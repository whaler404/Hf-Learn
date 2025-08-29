```python
for update in range(1, args.num_total_batches + 1):
    self.state.episode += 1 * args.batch_size
    data = next(iter_dataloader)
    with torch.no_grad():
        queries = data["input_ids"].to(device)
        context_length = queries.shape[1]
        responses = []
        postprocessed_responses = []
        logprobs = []
        ref_logprobs = []
        scores = []
        sequence_lengths = []
        values = []
        with unwrap_model_for_generation(
            self.model, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
        ) as unwrapped_model:
            query_responses, logitss = batch_generation(
                unwrapped_model.policy,
                queries,
                args.local_rollout_forward_batch_size,
                processing_class.pad_token_id,
                generation_config,
            )

        for i in range(0, queries.shape[0], args.local_rollout_forward_batch_size):
            query = queries[i : i + args.local_rollout_forward_batch_size]
            query_response = query_responses[i : i + args.local_rollout_forward_batch_size]
            response = query_response[:, context_length:]
            logits = logitss[i : i + args.local_rollout_forward_batch_size]
            logprob = selective_log_softmax(logits, response)
            del logits
            empty_cache()

            if ref_policy is None:
                with self.null_ref_context():
                    ref_output = forward(model.policy, query_response, processing_class.pad_token_id)
            else:
                ref_output = forward(ref_policy, query_response, processing_class.pad_token_id)
            ref_logits = ref_output.logits[:, context_length - 1 : -1]
            ref_logits /= args.temperature + 1e-7
            ref_logprob = selective_log_softmax(ref_logits, response)
            del ref_output, ref_logits
            empty_cache()

            # Response Processing 1. truncate response after the first occurrence of `stop_token_id`
            postprocessed_response = response
            if self.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                postprocessed_response = truncate_response(
                    self.stop_token_id, processing_class.pad_token_id, response
                )

            # Response Processing 2. run reward model on the truncated responses
            postprocessed_query_response = torch.cat((query, postprocessed_response), 1)
            sequence_length = first_true_indices(postprocessed_response == processing_class.pad_token_id) - 1
            unwrapped_value_model = accelerator.unwrap_model(model).value_model
            full_value, _, _ = get_reward(
                unwrapped_value_model, query_response, processing_class.pad_token_id, context_length
            )
            value = full_value[:, context_length - 1 : -1].squeeze(-1)
            _, score, _ = get_reward(
                reward_model, postprocessed_query_response, processing_class.pad_token_id, context_length
            )

            responses.append(response)
            postprocessed_responses.append(postprocessed_response)
            logprobs.append(logprob)
            ref_logprobs.append(ref_logprob)
            sequence_lengths.append(sequence_length)
            scores.append(score)
            values.append(value)
        responses = torch.cat(responses, 0)
        postprocessed_responses = torch.cat(postprocessed_responses, 0)
        logprobs = torch.cat(logprobs, 0)
        ref_logprobs = torch.cat(ref_logprobs, 0)
        sequence_lengths = torch.cat(sequence_lengths, 0)
        scores = torch.cat(scores, 0)
        values = torch.cat(values, 0)
        del (logprob, ref_logprob, full_value, value, score, unwrapped_model)
        empty_cache()
        gc.collect()

        # Response Processing 3. Filter completion. Ensure that the sample contains stop_token_id
        # Completions not passing that filter will receive a lower score.
        contain_eos_token = torch.any(postprocessed_responses == self.processing_class.eos_token_id, dim=-1)
        if self.args.missing_eos_penalty is not None:
            scores[~contain_eos_token] -= self.args.missing_eos_penalty
        # accelerator.print(f"{scores=}, {(contain_eos_token.sum() / len(contain_eos_token))=}")

        # be very careful with `padding_mask_p1`; see https://excalidraw.com/#json=LWnzG4w2k5DjF_EOL_xPt,e2w3a-hFJ_gX5vOfeyXGTw
        response_idxs = torch.arange(responses.shape[1], device=responses.device).repeat(responses.shape[0], 1)
        padding_mask = response_idxs > sequence_lengths.unsqueeze(1)
        logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
        ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)
        sequence_lengths_p1 = sequence_lengths + 1
        padding_mask_p1 = response_idxs > (sequence_lengths_p1.unsqueeze(1))
        values = torch.masked_fill(values, padding_mask_p1, 0)

        # 4. compute rewards
        # Formula used by http://joschu.net/blog/kl-approx.html for the k1 and k3 estimators
        logr = ref_logprobs - logprobs
        kl = -logr if args.kl_estimator == "k1" else (logr.exp() - 1) - logr  # Else statement is k3
        non_score_reward = -args.kl_coef * kl
        rewards = non_score_reward.clone()
        actual_start = torch.arange(rewards.size(0), device=rewards.device)
        actual_end = torch.where(sequence_lengths_p1 < rewards.size(1), sequence_lengths_p1, sequence_lengths)
        rewards[[actual_start, actual_end]] += scores

        # 5. whiten rewards
        if args.whiten_rewards:
            rewards = masked_whiten(rewards, mask=~padding_mask_p1, shift_mean=False)
            rewards = torch.masked_fill(rewards, padding_mask_p1, 0)

        # 6. compute advantages and returns
        lastgaelam = 0
        advantages_reversed = []
        gen_length = responses.shape[1]
        for t in reversed(range(gen_length)):
            nextvalues = values[:, t + 1] if t < gen_length - 1 else 0.0
            delta = rewards[:, t] + args.gamma * nextvalues - values[:, t]
            lastgaelam = delta + args.gamma * args.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], axis=1)
        returns = advantages + values
        advantages = masked_whiten(advantages, ~padding_mask)
        advantages = torch.masked_fill(advantages, padding_mask, 0)
        empty_cache()

    # Do multiple epochs of PPO training, with a fresh random shuffle in each epoch
    for ppo_epoch_idx in range(args.num_ppo_epochs):
        b_inds = np.random.permutation(args.local_batch_size)
        minibatch_idx = 0
        for mini_batch_start in range(0, args.local_batch_size, args.local_mini_batch_size):
            mini_batch_end = mini_batch_start + args.local_mini_batch_size
            mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
            gradient_accumulation_idx = 0
            for micro_batch_start in range(0, args.local_mini_batch_size, args.per_device_train_batch_size):
                with accelerator.accumulate(model):
                    micro_batch_end = micro_batch_start + args.per_device_train_batch_size
                    micro_batch_inds = mini_batch_inds[micro_batch_start:micro_batch_end]
                    mb_advantage = advantages[micro_batch_inds]
                    mb_responses = responses[micro_batch_inds]
                    mb_query_responses = query_responses[micro_batch_inds]
                    mb_logprobs = logprobs[micro_batch_inds]
                    mb_return = returns[micro_batch_inds]
                    mb_values = values[micro_batch_inds]

                    output, vpred_temp = forward(model, mb_query_responses, processing_class.pad_token_id)
                    logits = output.logits[:, context_length - 1 : -1]
                    logits /= args.temperature + 1e-7
                    new_logprobs = selective_log_softmax(logits, mb_responses)
                    new_logprobs = torch.masked_fill(
                        new_logprobs, padding_mask[micro_batch_inds], INVALID_LOGPROB
                    )
                    vpred = vpred_temp[:, context_length - 1 : -1].squeeze(-1)
                    vpred = torch.masked_fill(vpred, padding_mask_p1[micro_batch_inds], 0)
                    vpredclipped = torch.clamp(
                        vpred,
                        mb_values - args.cliprange_value,
                        mb_values + args.cliprange_value,
                    )
                    vf_losses1 = torch.square(vpred - mb_return)
                    vf_losses2 = torch.square(vpredclipped - mb_return)
                    vf_loss_max = torch.max(vf_losses1, vf_losses2)
                    vf_loss = 0.5 * masked_mean(vf_loss_max, ~padding_mask_p1[micro_batch_inds])
                    vf_clipfrac = masked_mean(
                        (vf_losses2 > vf_losses1).float(), ~padding_mask_p1[micro_batch_inds]
                    )
                    logprobs_diff = new_logprobs - mb_logprobs
                    ratio = torch.exp(logprobs_diff)
                    pg_losses = -mb_advantage * ratio
                    pg_losses2 = -mb_advantage * torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)
                    pg_loss_max = torch.max(pg_losses, pg_losses2)
                    pg_loss = masked_mean(pg_loss_max, ~padding_mask[micro_batch_inds])
                    loss = pg_loss + args.vf_coef * vf_loss
                    accelerator.backward(loss)
                    optimizer.step()
                    optimizer.zero_grad()
                    with torch.no_grad():
                        pg_clipfrac = masked_mean(
                            (pg_losses2 > pg_losses).float(), ~padding_mask[micro_batch_inds]
                        )
                        prob_dist = torch.nn.functional.softmax(logits, dim=-1)
                        entropy = torch.logsumexp(logits, dim=-1) - torch.sum(prob_dist * logits, dim=-1)
                        approxkl = 0.5 * (logprobs_diff**2).mean()
                        approxkl_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = approxkl
                        pg_clipfrac_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                            pg_clipfrac
                        )
                        pg_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = pg_loss
                        vf_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = vf_loss
                        vf_clipfrac_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                            vf_clipfrac
                        )
                        entropy_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = entropy.mean()
                        ratio_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = ratio.mean()
                gradient_accumulation_idx += 1
            minibatch_idx += 1

    self.lr_scheduler.step()
```
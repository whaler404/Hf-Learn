from distilabel.models import TransformersLLM, OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.steps import (
    LoadDataFromHub,
    GroupColumns,
    FormatTextGenerationDPO,
    PreferenceToArgilla,
)
from distilabel.steps.tasks import TextGeneration, UltraFeedback

import os

# enable visibility of cuda, only see cuda:4
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"

api_key = os.getenv("OPENROUTER_API_KEY")
base_url = "https://openrouter.ai/api/v1"

from huggingface_hub import login

login(token=os.getenv("HF_TOKEN"))

# load_dataset = LoadDataFromHub(
#         repo_id= "argilla/10Kprompts-mini",
#         num_examples=1,
#         pipeline=Pipeline(name="showcase-pipeline"),
#     )
# load_dataset.load()
# result = next(load_dataset.process())
# print("Loaded dataset:", result)

# # Loaded dataset: 
# # {
# #     "instruction": "How can I create an efficient and robust workflow that utilizes advanced automation techniques to extract targeted data, including customer information, from diverse PDF documents and effortlessly integrate it into a designated Google Sheet? Furthermore, I am interested in establishing a comprehensive and seamless system that promptly activates an SMS notification on my mobile device whenever a new PDF document is uploaded to the Google Sheet, ensuring real-time updates and enhanced accessibility.",
# #     "topic": "Software Development"
# # }


# generate_responses = [
#     TextGeneration(
#         # llm=TransformersLLM(
#         #     model="meta-llama/Llama-3.2-1B-Instruct",
#         #     generation_kwargs={"max_new_tokens": 512, "temperature": 0.7},
#         #     trust_remote_code=True,
#         # ),
#         llm=OpenAILLM(
#             base_url=base_url,
#             api_key=api_key,
#             model="meta-llama/llama-3.1-8b-instruct",
#             generation_kwargs={"max_new_tokens": 512, "temperature": 0.7},
#             max_retries=50,
#         ),
#         pipeline=Pipeline(name="showcase-pipeline"),
#     ),
#     TextGeneration(
#         # llm=TransformersLLM(
#         #     model="google/gemma-3-1b-it",
#         #     generation_kwargs={"max_new_tokens": 512, "temperature": 0.7},
#         #     trust_remote_code=True,
#         # ),
#         llm=OpenAILLM(
#             base_url=base_url,
#             api_key=api_key,
#             model="qwen/qwen-2.5-7b-instruct",
#             generation_kwargs={"max_new_tokens": 512, "temperature": 0.7},
#             max_retries=50,
#         ),
#         pipeline=Pipeline(name="showcase-pipeline"),
#     ),
# ]
# for task in generate_responses:
#     task.load()
#     print(next(task.process([{"instruction": "Which are the top cities in Spain?"}])))

# # [
# #   {
# #     "instruction": "Which are the top cities in Spain?",
# #     "generation": "Spain is a beautiful country with a rich culture and history, and it has many fascinating cities to explore. Here are some of the top cities in Spain:\n\n1. **Madrid**: The capital city of Spain, known for its vibrant nightlife, world-class museums (such as the Prado), and historic landmarks like the Royal Palace and Plaza Mayor.\n2. **Barcelona**: The capital of Catalonia, famous for its stunning architecture (Gaudí's Sagrada Família, Park Güell, and Casa Batlló), beaches, and lively nightlife.\n3. **Seville**: The charming capital of Andalusia, known for its Moorish architecture, beautiful parks and gardens, and flamenco music and dance.\n4. **Valencia**: A coastal city in eastern Spain, famous for its City of Arts and Sciences complex, beautiful beaches, and traditional cuisine (paella).\n5. **Granada**: A historic city in southern Spain, known for its breathtaking Alhambra palace and Generalife gardens, a UNESCO World Heritage Site.\n6. **San Sebastián**: A beautiful coastal city in the Basque Country, famous for its gastronomy (pintxos), beautiful beaches, and vibrant cultural scene.\n7. **Zaragoza**: The capital of Aragon, known for its Roman and Moorish history, beautiful architecture, and vibrant cultural scene.\n8. **Alicante**: A popular coastal city in southeastern Spain, known for its beautiful beaches, historic old town, and scenic coastal promenade.\n9. **Bilbao**: A city in the Basque Country, famous for its Guggenheim Museum, beautiful architecture, and stunning natural surroundings.\n10. **Córdoba**: A historic city in Andalusia, known for its impressive Mezquita (Mosque-Cathedral) and beautiful Roman and Moorish architecture.\n\nThese are just a few of the many amazing cities in Spain, and there are many more to discover. Each city has its own unique character, history, and culture, making Spain a fascinating country to explore.\n\nHonorable mentions:\n\n* **Málaga**: A coastal city in the Costa del Sol region, known for its beautiful beaches, historic center, and Picasso Museum.\n* **València**: A city in eastern Spain, known for its City of Arts and Sciences complex and traditional cuisine.\n* **Gijón**: A coastal city in northern Spain, famous for its beautiful beaches, historic center, and vibrant cultural scene.\n* **Santander**: A",
# #     "distilabel_metadata": {
# #       "raw_output_text_generation_0": "Spain is a beautiful country with a rich culture and history, and it has many fascinating cities to explore. Here are some of the top cities in Spain:\n\n1. **Madrid**: The capital city of Spain, known for its vibrant nightlife, world-class museums (such as the Prado), and historic landmarks like the Royal Palace and Plaza Mayor.\n2. **Barcelona**: The capital of Catalonia, famous for its stunning architecture (Gaudí's Sagrada Família, Park Güell, and Casa Batlló), beaches, and lively nightlife.\n3. **Seville**: The charming capital of Andalusia, known for its Moorish architecture, beautiful parks and gardens, and flamenco music and dance.\n4. **Valencia**: A coastal city in eastern Spain, famous for its City of Arts and Sciences complex, beautiful beaches, and traditional cuisine (paella).\n5. **Granada**: A historic city in southern Spain, known for its breathtaking Alhambra palace and Generalife gardens, a UNESCO World Heritage Site.\n6. **San Sebastián**: A beautiful coastal city in the Basque Country, famous for its gastronomy (pintxos), beautiful beaches, and vibrant cultural scene.\n7. **Zaragoza**: The capital of Aragon, known for its Roman and Moorish history, beautiful architecture, and vibrant cultural scene.\n8. **Alicante**: A popular coastal city in southeastern Spain, known for its beautiful beaches, historic old town, and scenic coastal promenade.\n9. **Bilbao**: A city in the Basque Country, famous for its Guggenheim Museum, beautiful architecture, and stunning natural surroundings.\n10. **Córdoba**: A historic city in Andalusia, known for its impressive Mezquita (Mosque-Cathedral) and beautiful Roman and Moorish architecture.\n\nThese are just a few of the many amazing cities in Spain, and there are many more to discover. Each city has its own unique character, history, and culture, making Spain a fascinating country to explore.\n\nHonorable mentions:\n\n* **Málaga**: A coastal city in the Costa del Sol region, known for its beautiful beaches, historic center, and Picasso Museum.\n* **València**: A city in eastern Spain, known for its City of Arts and Sciences complex and traditional cuisine.\n* **Gijón**: A coastal city in northern Spain, famous for its beautiful beaches, historic center, and vibrant cultural scene.\n* **Santander**: A",
# #       "raw_input_text_generation_0": [
# #         {
# #           "role": "user",
# #           "content": "Which are the top cities in Spain?"
# #         }
# #       ],
# #       "statistics_text_generation_0": {
# #         "input_tokens": 19,
# #         "output_tokens": 512
# #       }
# #     },
# #     "model_name": "meta-llama/llama-3.1-8b-instruct"
# #   }
# # ]
# # [
# #   {
# #     "instruction": "Which are the top cities in Spain?",
# #     "generation": "Spain has a diverse range of cities, each with its own unique characteristics. Here are some of the top cities in Spain, based on factors like culture, tourism, economy, and quality of life:\n\n1. **Barcelona**: Known for its vibrant culture, architecture, and beaches. It's home to famous landmarks like the Sagrada Familia and Park Güell, as well as a bustling nightlife and food scene.\n\n2. **Madrid**: The capital city is a hub of politics, art, and commerce. It's home to La Gran Vía, Prado Museum, and Retiro Park. Madrid is also known for its nightlife and wide variety of culinary options.\n\n3. **Seville**: Famous for its flamenco culture, beautiful architecture, and delicious cuisine. The city is also known for its historic Alcázar palace and the famous Seville Cathedral.\n\n4. **Valencia**: Known for its beautiful beaches, excellent food, and innovative architecture. The City of Arts and Sciences is a must-visit attraction.\n\n5. **Malaga**: A city with a beautiful coastline, rich history, and vibrant nightlife. The city is home to the famous Picasso Museum and the Alcazaba fortress.\n\n6. **Granada**: Known for its stunning Alhambra palace and beautiful historic center. The city is also famous for its vibrant nightlife and delicious cuisine.\n\n7. **Santander**: A city with a beautiful coastline and a lively arts scene. It's known for its architecture, such as the Palacio de Festivales, and its excellent beaches.\n\n8. **San Sebastian**: A small city on the Basque coast known for its delicious cuisine, beautiful beaches, and vibrant nightlife. It's also home to the famous Guggenheim Museum.\n\n9. **Alicante**: A coastal city with beautiful beaches, a vibrant nightlife, and a rich cultural heritage. The city is also known for its delicious seafood and nightlife.\n\n10. **Valencia**: Known for its beautiful beaches, excellent food, and innovative architecture. The City of Arts and Sciences is a must-visit attraction.\n\nThese cities offer a wide range of experiences, from historic landmarks to modern architecture, beautiful beaches to vibrant nightlife, and delicious cuisine to cultural events.",
# #     "distilabel_metadata": {
# #       "raw_output_text_generation_0": "Spain has a diverse range of cities, each with its own unique characteristics. Here are some of the top cities in Spain, based on factors like culture, tourism, economy, and quality of life:\n\n1. **Barcelona**: Known for its vibrant culture, architecture, and beaches. It's home to famous landmarks like the Sagrada Familia and Park Güell, as well as a bustling nightlife and food scene.\n\n2. **Madrid**: The capital city is a hub of politics, art, and commerce. It's home to La Gran Vía, Prado Museum, and Retiro Park. Madrid is also known for its nightlife and wide variety of culinary options.\n\n3. **Seville**: Famous for its flamenco culture, beautiful architecture, and delicious cuisine. The city is also known for its historic Alcázar palace and the famous Seville Cathedral.\n\n4. **Valencia**: Known for its beautiful beaches, excellent food, and innovative architecture. The City of Arts and Sciences is a must-visit attraction.\n\n5. **Malaga**: A city with a beautiful coastline, rich history, and vibrant nightlife. The city is home to the famous Picasso Museum and the Alcazaba fortress.\n\n6. **Granada**: Known for its stunning Alhambra palace and beautiful historic center. The city is also famous for its vibrant nightlife and delicious cuisine.\n\n7. **Santander**: A city with a beautiful coastline and a lively arts scene. It's known for its architecture, such as the Palacio de Festivales, and its excellent beaches.\n\n8. **San Sebastian**: A small city on the Basque coast known for its delicious cuisine, beautiful beaches, and vibrant nightlife. It's also home to the famous Guggenheim Museum.\n\n9. **Alicante**: A coastal city with beautiful beaches, a vibrant nightlife, and a rich cultural heritage. The city is also known for its delicious seafood and nightlife.\n\n10. **Valencia**: Known for its beautiful beaches, excellent food, and innovative architecture. The City of Arts and Sciences is a must-visit attraction.\n\nThese cities offer a wide range of experiences, from historic landmarks to modern architecture, beautiful beaches to vibrant nightlife, and delicious cuisine to cultural events.",
# #       "raw_input_text_generation_0": [
# #         {
# #           "role": "user",
# #           "content": "Which are the top cities in Spain?"
# #         }
# #       ],
# #       "statistics_text_generation_0": {
# #         "input_tokens": 16,
# #         "output_tokens": 456
# #       }
# #     },
# #     "model_name": "qwen/qwen-2.5-7b-instruct"
# #   }
# # ]

# group_responses = GroupColumns(
#     columns=["generation", "model_name"],
#     output_columns=["generations", "model_names"],
#     pipeline=Pipeline(name="showcase-pipeline"),
# )
# result = next(
#     group_responses.process(
#         [
#             {
#                 "generation": "Madrid",
#                 "model_name": "meta-llama/llama-3.1-8b-instruct",
#             },
#         ],
#         [
#             {
#                 "generation": "Barcelona",
#                 "model_name": "qwen/qwen-2.5-7b-instruct",
#             }
#         ],
#     )
# )
# print("Grouped responses:", result)

# # Grouped responses:
# # [
# #   {
# #     "generations": [
# #       "Madrid",
# #       "Barcelona"
# #     ],
# #     "model_names": [
# #       "meta-llama/llama-3.1-8b-instruct",
# #       "qwen/qwen-2.5-7b-instruct"
# #     ]
# #   }
# # ]


# evaluate_responses = UltraFeedback(
#     aspect="overall-rating",
#     # llm=TransformersLLM(
#     #     model="Qwen/Qwen3-4B",
#     #     generation_kwargs={"max_new_tokens": 512, "temperature": 0.7},
#     #     trust_remote_code=True,
#     # ),
#     llm=OpenAILLM(
#         base_url=base_url,
#         api_key=api_key,
#         model="meta-llama/llama-3.1-70b-instruct",
#         generation_kwargs={"max_new_tokens": 512, "temperature": 0.7},
#         max_retries=50,
#     ),
#     pipeline=Pipeline(name="showcase-pipeline"),
# )
# evaluate_responses.load()
# result = next(
#     evaluate_responses.process(
#         [
#             {
#                 "instruction": "What's the capital of Spain?",
#                 "generations": ["Madrid", "Barcelona"],
#             }
#         ]
#     )
# )
# print("Evaluated responses:", result)

# # Evaluated responses:
# # {
# #     "instruction": "What's the capital of Spain?",
# #     "generations": [
# #         {
# #             "text": "Madrid",
# #             "model_name": "meta-llama/llama-3.1-70b-instruct",
# #             "rating": 5,
# #             "rationale": "The answer is accurate, directly addresses the question, and shows confidence in providing the correct information. It aligns perfectly with the instruction and does not introduce any hallucinations."
# #         },
# #         {
# #             "text": "Barcelona",
# #             "model_name": "qwen/qwen-2.5-7b-instruct",
# #             "rating": 1,
# #             "rationale": "The answer is incorrect, as Barcelona is not the capital of Spain. It fails to address the question accurately and introduces misleading information, which could be considered a hallucination."
# #         }
# #     ],
# #     "metadata": {
# #         "raw_output_ultra_feedback": [
# #             {
# #                 "text": "Madrid",
# #                 "rating": 5,
# #                 "rationale": "The answer is accurate, directly addresses the question, and shows confidence in providing the correct information. It aligns perfectly with the instruction and does not introduce any hallucinations."
# #             },
# #             {
# #                 "text": "Barcelona",
# #                 "rating": 1,
# #                 "rationale": "The answer is incorrect, as Barcelona is not the capital of Spain. It fails to address the question accurately and introduces misleading information, which could be considered a hallucination."
# #             }
# #         ],
# #         "raw_input_ultra_feedback": [
# #             {
# #                 "role": "system",
# #                 "content": "Your role is to evaluate text quality based on given criteria.\nYou'll receive an instructional description (\"Instruction\") and 2 text outputs (\"Text\").\nUnderstand and interpret instructions to evaluate effectively.\nProvide annotations for each text with a rating and rationale.\nThe 2 texts given are independent, and should be evaluated separately."
# #             },
# #             {
# #                 "role": "user",
# #                 "content": "# General Text Quality Assessment\n\nEvaluate the model's outputs based on various criteria:\n\n1. **Correctness & Informativeness**: Does the output provide accurate and helpful information?\n2. **Honesty & Uncertainty**: How confidently does the model convey its information, and does it express uncertainty appropriately?\n3. **Truthfulness & Hallucination**: Does the model introduce misleading or fabricated details?\n4. **Instruction Following**: Does the model's output align with given instructions and the user's intent?\n\nYour role is to provide a holistic assessment considering all the above factors.\n\n**Scoring**: Rate outputs 1 to 5 based on the overall quality, considering all aspects:\n1. **Low Quality**: Contains inaccuracies, may be entirely wrong or has severe hallucinations.\n2. **Moderate Quality**: Addresses some aspects, but has errors or is partially aligned with instructions.\n3. **Good**: Generally accurate but may contain minor errors or slight deviations.\n4. **Very Good**: Near perfect, with minor issues in terms of alignment or confidence.\n5, **Excellent**: Accurate, confident, aligned with instructions, and free of hallucinations."
# #             }
# #         ],
# #         "statistics_ultra_feedback": {
# #             "input_tokens": 436,
# #             "output_tokens": 103
# #         }
# #     }
# # }

# format_dpo = FormatTextGenerationDPO(pipeline=Pipeline(name="showcase-pipeline"))
# format_dpo.load()
# result = next(
#     format_dpo.process(
#         [
#             {
#                 "instruction": "What's the capital of Spain?",
#                 "generations": ["Madrid", "Barcelona"],
#                 "generation_models": [
#                     "meta-llama/Llama-3.2-1B-Instruct",
#                     "qwen/qwen-2.5-7b-instruct",
#                 ],
#                 "ratings": [5, 1],
#             }
#         ]
#     )
# )
# print("Formatted for DPO training:", result)

# # Formatted for DPO training: 
# # {
# #   "instruction": "What's the capital of Spain?",
# #   "generations": [
# #     {
# #       "content": "Madrid",
# #       "model": "meta-llama/Llama-3.2-1B-Instruct",
# #       "rating": 5
# #     },
# #     {
# #       "content": "Barcelona",
# #       "model": "qwen/qwen-2.5-7b-instruct",
# #       "rating": 1
# #     }
# #   ],
# #   "prompt": {
# #     "content": "What's the capital of Spain?",
# #     "id": "26174c953df26b3049484e4721102dca6b25d2de9e3aa22aa84f25ed1c798512"
# #   },
# #   "chosen": {
# #     "dialogue": [
# #       {
# #         "role": "user",
# #         "content": "What's the capital of Spain?"
# #       },
# #       {
# #         "role": "assistant",
# #         "content": "Madrid"
# #       }
# #     ],
# #     "model": "meta-llama/Llama-3.2-1B-Instruct",
# #     "rating": 5
# #   },
# #   "rejected": {
# #     "dialogue": [
# #       {
# #         "role": "user",
# #         "content": "What's the capital of Spain?"
# #       },
# #       {
# #         "role": "assistant",
# #         "content": "Barcelona"
# #       }
# #     ],
# #     "model": "qwen/qwen-2.5-7b-instruct",
# #     "rating": 1
# #   }
# # }


with Pipeline(name="generate-dataset") as pipeline:

    load_dataset = LoadDataFromHub(repo_id="argilla/10Kprompts-mini")

    generate_responses = [
        TextGeneration(
            llm=OpenAILLM(
                base_url=base_url,
                api_key=api_key,
                model="meta-llama/llama-3.1-8b-instruct",
                generation_kwargs={"max_new_tokens": 512, "temperature": 0.7},
                max_retries=50,
            )
        ),
        TextGeneration(
            llm=OpenAILLM(
                base_url=base_url,
                api_key=api_key,
                model="qwen/qwen-2.5-7b-instruct",
                generation_kwargs={"max_new_tokens": 512, "temperature": 0.7},
                max_retries=50,
            )
        ),
    ]

    group_responses = GroupColumns(
        columns=["generation", "model_name"],
        output_columns=["generations", "model_names"],
    )

    evaluate_responses = UltraFeedback(
        aspect="overall-rating",
        llm=OpenAILLM(
            base_url=base_url,
            api_key=api_key,
            # model="meta-llama/llama-3.1-70b-instruct",
            model="qwen/qwen-2.5-7b-instruct",
            generation_kwargs={"max_new_tokens": 512, "temperature": 0.7},
            max_retries=50,
        )
    )

    format_dpo = FormatTextGenerationDPO()

    to_argilla = PreferenceToArgilla(
        dataset_name="preference-dataset",
        dataset_workspace="argilla",
        api_url="https://wheeler404-my-argilla.hf.space",
        api_key="w7UXPkY3kEZePcS9x1nia271CA2HuXak46YqzLYz2WhwGcTJKw8dMbXzPZTYapkl39y6yb3XY5RvHjDIqKqVUAbalVwSZsQUAV5XP_9eayw",
        num_generations=2
    )

    # orchestration
    for task in generate_responses:
        load_dataset.connect(task)
        task.connect(group_responses)
    group_responses.connect(evaluate_responses)
    evaluate_responses.connect(format_dpo, to_argilla)

distiset = pipeline.run(use_cache=False)

# distiset.push_to_hub("wheeler404/example-preference-dataset")

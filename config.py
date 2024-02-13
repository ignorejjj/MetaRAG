import os

os.environ["OPENAI_API_KEY"] = "your-api-key"

# dataset path
DATASET2PATH = {
    "bamboogle": "./data/bamboogle.json",
    "musique": "./data/musique.json",
    "hotpotqa": "./data/hotpotqa.json",
    "2wiki": "./data/2WikiMultihopQA.json"
}
# for 2wiki dataset
ID_ALIASES_PATH = "./data/2wiki_id_aliases.json"
QUERY_PATH = "./data/2wiki_queries.jsonl"

# path for retriever
INDEX_PATH = None  # you can change it to your own build index path
RERANKER_PATH = "intfloat/simlm-msmarco-reranker"

# path for critic
NLI_MODEL_PATH = "google/t5_xxl_true_nli_mixture"

# path for llm
LLAMA2_13B_CHAT_PATH = "meta-llama/Llama-2-13b-chat-hf"
CHATGLM2_PATH = "THUDM/chatglm2-6b"
VACUNA_PATH = "lmsys/vicuna-13b-v1.5"

# path for monitor
EXPERT_MODEL_PATH = {"span-bert": "haritzpuerto/spanbert-large-cased_HotpotQA",
                     "t5": "gaussalgo/T5-LM-Large_Canard-HotpotQA-rephrase",
                    "llama2": LLAMA2_13B_CHAT_PATH,
                    "chatglm2": CHATGLM2_PATH,
}

SIMILARITY_MODEL_PATH = "sentence-transformers/all-MiniLM-L6-v2"


# set prompt templete
QA_PROMPT = """Answer the question by thinking step by step. In your reasoning process, you need to carefully search for the information you need in the reference. If you need additional information, give the query you want to search.
                       Reference: {reference}
                       Question: {question}
                       {all_hint}
                       Your output MUST follow the following format:
                       ```json 
                       {{"reason": "your reasoning steps","answer": "your final answer in ONE OR FEW WORDS"}}
                       ```
                       Here are some examples:
                        # 
                        Reference: John, Count Palatine of Neumarkt: John (Johann von Pfalz-Neumarkt; 1383 – 14 March 1443) was the Count
                        Palatine of Neumarkt from 1410 to his death. The son of Rupert III of the Palatinate, he married Catherine of Pomerania in
                        1407.
                        Question: Who is Catherine Of Pomerania, Countess Palatine Of Neumarkt’s father-in-law?
                        Expected Output:
                        ```json
                        {{"reason": "The husband of Catherine of Pomerania, Countess Palatine of Neumarkt is John, Count Palatine of
                        Neumarkt. The father of John, Count Palatine of Neumarkt is Rupert III of the Palatinate. So the final answer is: Rupert III of the Palatinate.",
                        "answer": "Rupert III of the Palatinate"}}
                        ```
                        #
                        Reference: Crimen a las tres: Crimen a las tres is a 1935 Argentine crime film directed and written by Luis Saslavsky. Crimen
                        a las tres. Directed by, Luis Saslavsky. Elio Petri: The Working Class Goes to Heaven (Italian: La classe operaia va in paradiso), released in the US as
                        Lulu the Tool, is a 1971 political drama film directed by Elio Petri. March 20, 1995: Luis Saslavsky (April 21, 1903 – March 20, 1995) was an Argentine film director, screenwriter
                        and film producer, and one of the influential directors in the Cinema of Argentina of the classic era. Elio Petri: Final years. In 1981, Petri visited Geneva to direct Arthur Millers new play The American Clock, with ´
                        Marcello Mastroianni playing the lead role. Petri died of cancer on 10 November 1982.
                        Question: Which film has the director died first, Crimen A Las Tres or The Working Class Goes To Heaven?
                        Expected Output:
                        ```json
                        {{"reason":"The director of Crimen a las tres is Luis Saslavsky.The director of The Working Class Goes to Heaven is Elio Petri. Luis Saslavsky died on March 20, 1995.Elio Petri died on 10 November 1982. So the director of The Working Class Goes to Heaven died first.",
                        "answer":"The Working Class Goes to Heaven"}}
                        ```
                        #
                        Reference: Indogrammodes is a genus of moths of the Crambidae family.It contains only one species, Indogrammodes pectinicornalis, which is found in India. India, officially the Republic of India ("Bhārat Gaṇarājya"), is a country in South Asia. It is the seventh-largest country by area, the second-most populous country (with over 1.2 billion people), and the most populous democracy in the world.
                        Question: Which genus of moth in the world's seventh-largest country contains only one species?
                        Expected Output:
                        ```json
                        {{"reason": "The world's seventh-largest country is India. Indogrammodes contains only one species and is found in India, and the genus of Indogrammodes is Crambidae.So the final answer is: Crambidae.",
                        "answer": "Crambidae"}}
                        ```
                        # 
                        Reference: The 2013 Liqui Moly Bathurst 12 Hour was an endurance race for a variety of GT and touring car classes, including: GT3 cars, GT4 cars, Group 3E Series Production Cars and Dubai 24 Hour cars. The event, which was staged at the Mount Panorama Circuit, near Bathurst, in New South Wales, Australia on 10 February 2013, was the eleventh running of the Bathurst 12 Hour.Mount Panorama Circuit is a motor racing track located in Bathurst, New South Wales, Australia. The 6.213 km long track is technically a street circuit, and is a public road, with normal speed restrictions, when no racing events are being run, and there are many residences which can only be accessed from the circuit.
                        Question: What is the length of the track where the 2013 Liqui Moly Bathurst 12 Hour was staged?
                        Expected Output:
                        ```json
                        {{"reason": "The track where the 2013 Liqui Moly Bathurst 12 Hour was staged is Mount Panorama Circuit. And the length of Mount Panorama Circuit is 6.213 km long. So the final answer is: 6.213 km long.",
                        "answer": "6.213 km long"}}
                        ```
"""



CHECK_PROMPT = """
                    I need you to act as an inspector to check if there are any basic errors in the user's answer to the question. Below are some typical error types and corresponding examples.
                    Examples: 
                        1. Logic error
                        question: What director worked with Vikram Bhatt on a film starring actors Rajneesh Duggal and Adah Sharma?
                        user_answer: Vikram Bhatt
                        ground truth: Tinu Suresh Desai
                        explaination: The question is about someone who works with Vikram Bhatt, not Vikram Bhatt himself.
                        2. Not align with the question's requirement
                        question: The Waterloo Vase has been used as a garden ornament at whose residence and administrative headquarters?
                        user_answer: Buckingham Palace
                        ground truth: reigning monarch of the United Kingdom
                        explaination: The question is about  "whose residence", so the answer should refer to a person, not a place.
                        3. Answer redundancy
                        question: What type of film are both \"500 Years Later\" and \"Manson\"?
                        user_answer: Documentary films
                        ground truth: documentary
                        explaination: The question is about the type of film, just answer the film type, no need to follow "films".
                        4. Answer redundancy
                        question: Who is older, Jack Ma or Faye Wong?
                        user_answer: Jack Ma is older
                        ground truth: Jack Ma
                        explaination: Just answer the person's name, no additional explanation required.
                    Here is the user's answer that you need to check.
                    Question: {question}
                    user_answer: {answer}
                    Please check if the user's answer has made the above error and give your judgment and feedback. Your feedback can be similar to the explanation above.
                    Output in JSON format like:
                    ```json
                    {{"judgement":the error type(or correct),"feedback":some short reminders you would like to give}}
                    ```
                    """

REWRITE_PROMPT = """Question: {question}
            Reference: {reference}
            Given answer: {answer}
            A person answered a question based on a reference information. However, due to issues with the reference (lack of information), more references need to be retrieved now. Please carefully consider his answers and questions step by step, and provide a query for searching based on the information he needs to answer the question.
            Your output MUST with the ending of "The rewrite query is ... 
            """

EXTERNAL_PROMPT = "I need you to help me determine if the provided reference can answer the question, you just need to answer yes or no.\n\nQuestion: {question}Reference: {reference}\n\nYou just need to answer yes or no.\nAnswer:"
INTERNAL_PROMPT = "Determine if you can provide a reliable answer to the question: '{question}' based on your own knowledge(output yes or no)?"
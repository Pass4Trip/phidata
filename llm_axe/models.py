from ollama import Client
import openai

class OllamaChat():
    def __init__(self, host:str="http://localhost:11434", model:str=None):

        if model is None:
            raise ValueError('''You must provide a model to use OllamaChat. 
                                example: OllamaChat(model='llama3:instruct')''')

        self._host = host
        self._model = model
        self._ollama = Client(host)

    def ask(self, prompts:list, format:str="", temperature:float=0.8, stream:bool=False, **options):
        """
        Args:
            prompts (list): A list of prompts to ask.
            format (str, optional): The format of the response. Use "json" for json. Defaults to "".
            temperature (float, optional): The temperature of the LLM. Defaults to 0.8.
        """
        if stream is True:
            return self._ollama.chat(model=self._model, messages=prompts, format=format, options={"temperature": temperature, **options}, stream=stream)
        return self._ollama.chat(model=self._model, messages=prompts, format=format, options={"temperature": temperature, **options}, stream=stream)["message"]["content"]        


class OpenAIChat():
    def __init__(self, api_key:str=None, model:str="gpt-4o-mini"):
        if not api_key:
            raise ValueError('''Vous devez fournir une clé API OpenAI valide. 
                                Exemple : OpenAIChat(api_key='votre_clé_api')''')

        self._model = model
        try:
            self._client = openai.OpenAI(api_key=api_key)
        except Exception as e:
            raise ValueError(f"Impossible de créer le client OpenAI : {e}")

    def ask(self, prompts:list, format:str="", temperature:float=0.8, stream:bool=False, **options):
        """
        Args:
            prompts (list): Une liste de messages pour le modèle.
            format (str, optional): Le format de réponse. Supporte 'json'.
            temperature (float, optional): La température du modèle. Par défaut à 0.8.
            stream (bool, optional): Si la réponse doit être streamée. Par défaut à False.
            **options: Options supplémentaires pour l'API OpenAI.
        """
        try:
            # Paramètres par défaut
            create_params = {
                "model": self._model,
                "messages": prompts,
                "temperature": temperature,
                "stream": stream
            }

            # Gestion du format JSON
            if format == "json":
                create_params["response_format"] = {"type": "json_object"}

            # Ajouter les options supplémentaires
            create_params.update(options)

            # Créer la complétion
            if stream:
                return self._client.chat.completions.create(**create_params)
            
            response = self._client.chat.completions.create(**create_params)
            
            # Retourner le contenu du message
            return response.choices[0].message.content
        
        except openai.AuthenticationError as e:
            print(f"Erreur d'authentification OpenAI : {e}")
            raise
        except openai.APIError as e:
            print(f"Erreur de l'API OpenAI : {e}")
            raise
        except Exception as e:
            print(f"Erreur inattendue lors de l'appel à l'API OpenAI : {e}")
            raise

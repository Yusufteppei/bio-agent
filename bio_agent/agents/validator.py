from typing import List
from bio_agent.agent import BioAgent


class Validator(BioAgent):

    def __init__(self, model_path):
        super().__init__(model_path)

        
    def validate_user_input(self):
        """
            Confirm it's a biological request
            
        """
        pass
        

    def validate_executor_output(self):
        """
            Implements access policies
            
        """
        pass
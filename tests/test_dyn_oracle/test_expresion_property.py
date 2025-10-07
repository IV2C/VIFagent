import unittest

import parameterized
from unittest.mock import MagicMock
from openai import Client
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from vif.falcon.oracle.guided_oracle.expressions import OracleExpression
from vif.falcon.oracle.guided_oracle.property_expression import visual_property


from PIL import Image

class TestExpressionProperty(unittest.TestCase):
    """Note: Does not test expressions which are property(), hence no need for llm instantiation"""

    def create_dummy_client_with_response(self,response:str):
        dummy_client = Client(base_url="NONE", api_key="NONE")
        dummy_model = "DUMMY_MODEL"
        dummy_response = ChatCompletion(
            id="mock",
            object="chat.completion",
            created=0,
            model=dummy_model,
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant", content=response
                    ),
                    finish_reason="stop",
                )
            ],
        )

        dummy_client.chat.completions.create = MagicMock(return_value=dummy_response)
        
        return dummy_client,dummy_model

    def test_property_valid(self):
        property = "The monkey is now smiling"

        dummy_text_response = "Lorem Ipsum Dolor Sit Amet\n fsqdfqsdfniaueb \\boxed{True}"


        original_image = Image.open("tests/resources/monkey.png")
        custom_image = Image.open("tests/resources/monkey_sad.png")


        dummy_client,dummy_model = self.create_dummy_client_with_response(dummy_text_response)
        
        
        def test_valid_customization() -> bool:
            return visual_property(property)
        expression: OracleExpression = test_valid_customization()
        result, feedback = expression.evaluate(
            original_image=original_image,
            custom_image=custom_image,
            client = dummy_client,
            model=dummy_model,
            temperature=0.0
        )
        self.assertTrue(result)
        self.assertEqual([], feedback)

    def test_property_invalid(self):
        property = "The monkey is now smiling"

        dummy_text_response = "Lorem Ipsum Dolor Sit Amet\n fsqdfqsdfniaueb \\boxed{False}"


        original_image = Image.open("tests/resources/monkey.png")
        custom_image = Image.open("tests/resources/monkey_sad.png")


        dummy_client,dummy_model = self.create_dummy_client_with_response(dummy_text_response)
        
        
        def test_valid_customization() -> bool:
            return visual_property(property)
        expression: OracleExpression = test_valid_customization()
        result, feedback = expression.evaluate(
            original_image=original_image,
            custom_image=custom_image,
            client = dummy_client,
            model=dummy_model,
            temperature=0.0
        )
        expected_feedback = f'The property "{property}" is not applied, but should be.'
        self.assertFalse(result)
        self.assertEqual([expected_feedback], feedback)

    def test_property_negated_valid(self):
        property = "The monkey is now smiling"

        dummy_text_response = "Lorem Ipsum Dolor Sit Amet\n\\boxed{False}"


        original_image = Image.open("tests/resources/monkey.png")
        custom_image = Image.open("tests/resources/monkey_sad.png")


        dummy_client,dummy_model = self.create_dummy_client_with_response(dummy_text_response)
        
        
        def test_valid_customization() -> bool:
            return ~visual_property(property)
        expression: OracleExpression = test_valid_customization()
        result, feedback = expression.evaluate(
            original_image=original_image,
            custom_image=custom_image,
            client = dummy_client,
            model=dummy_model,
            temperature=0.0
        )
        self.assertTrue(result)
        self.assertEqual([], feedback)

    def test_property_negated_invalid(self):
        property = "The monkey is now smiling"

        dummy_text_response = "Lorem Ipsum Dolor Sit Amet\n\\boxed{True}"


        original_image = Image.open("tests/resources/monkey.png")
        custom_image = Image.open("tests/resources/monkey_sad.png")


        dummy_client,dummy_model = self.create_dummy_client_with_response(dummy_text_response)
        
        
        def test_valid_customization() -> bool:
            return ~visual_property(property)
        expression: OracleExpression = test_valid_customization()
        result, feedback = expression.evaluate(
            original_image=original_image,
            custom_image=custom_image,
            client = dummy_client,
            model=dummy_model,
            temperature=0.0
        )
        expected_feedback = f'The property "{property}" is applied, but shouldn\'t be.'
        self.assertFalse(result)
        self.assertEqual([expected_feedback], feedback)
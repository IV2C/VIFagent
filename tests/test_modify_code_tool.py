import unittest

from vif.falcon.edition import OracleEditionModule


class TestModifyTool(unittest.TestCase):
    def test_edition_modify_simple(self):
        input = """
        The quick brown fox
        jumps over the lazy dog
        
        Lorem ipsum dolor sit amet
        consectetur adipiscing elit
        Sed do eiusmod tempor
        """

        edits: list = [
            {
                "start": 2,
                "end": 4,
                "content": "        The quick brown dog\n        jumps over the lazy cat",
            },
        ]

        edition_module = OracleEditionModule(
            model="",
            client=None,
        )
        expected = """
        The quick brown dog
        jumps over the lazy cat
        
        Lorem ipsum dolor sit amet
        consectetur adipiscing elit
        Sed do eiusmod tempor
        """
        expected_annotated = """1|
2|        The quick brown dog
3|        jumps over the lazy cat
4|        
5|        Lorem ipsum dolor sit amet
6|        consectetur adipiscing elit
7|        Sed do eiusmod tempor
8|        """

        edited, annotated = edition_module.modify_code(edits=edits, code=input)

        self.assertEqual(expected, edited)
        self.assertEqual(expected_annotated, annotated)

    def test_edition_multiple_non_contiguous(self):
        input = """
        A journey of a thousand miles
        begins with a single step
        
        To be or not to be
        that is the question
        
        Knowledge is power
        """
        edits = [
            {"start": 2, "end": 3, "content": "        A journey of ten miles"},
            {"start": 5, "end": 6, "content": "        To code or not to code"},
            {"start": 8, "end": 9, "content": "        Wisdom is power"},
        ]
        edition_module = OracleEditionModule(model="", client=None)
        expected = """
        A journey of ten miles
        begins with a single step
        
        To code or not to code
        that is the question
        
        Wisdom is power
        """
        edited,annotated = edition_module.modify_code(edits=edits, code=input)
        self.assertEqual(expected, edited)


    def test_edition_insertions_only(self):
        input = """
        Hello world
        """
        edits = [
            {"start": 2, "end": 2, "content": "        Greetings everyone"},
            {"start": 3, "end": 3, "content": "        Have a nice day"},
        ]
        edition_module = OracleEditionModule(model="", client=None)
        expected = """
        Greetings everyone
        Hello world
        Have a nice day
        """
        edited,annotated = edition_module.modify_code(edits=edits, code=input)
        self.assertEqual(expected, edited)


    def test_edition_deletions_only(self):
        input = """
        Line one
        Line two
        Line three
        Line four
        """
        edits = [
            {"start": 2, "end": 3, "content": ""},
            {"start": 4, "end": 6, "content": ""},
        ]
        edition_module = OracleEditionModule(model="", client=None)
        expected = """
        Line two
        """
        edited,annotated = edition_module.modify_code(edits=edits, code=input)
        self.assertEqual(expected, edited)


    def test_edition_overlapping_edits(self):
        input = """
        Apples are red
        Bananas are yellow
        Grapes are purple
        """
        edits = [
            {"start": 2, "end": 3, "content": "        Apples are green"},
            {"start": 3, "end": 5, "content": "        Bananas are green too"},
        ]
        edition_module = OracleEditionModule(model="", client=None)
        expected = """
        Apples are green
        Bananas are green too
        """
        edited,annotated = edition_module.modify_code(edits=edits, code=input)
        self.assertEqual(expected, edited)

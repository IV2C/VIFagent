import unittest

from vif.CodeMapper.feature import MappedCode
from vif.models.code import CodeEdit



class TestMappedCode(unittest.TestCase):

    def test_mappedcode_edit(self):
        # input
        code = """        dog
        cat
        
        foo
        bar
        foo
        """

        edits: list[CodeEdit] = [CodeEdit(2, 5, "        cat")]

        mapped_code = MappedCode(code=code, feature_map=None, image=None)

        # expected
        expected_code = """        dog
        cat
        bar
        foo
        """
        edited_code = mapped_code.apply_edits(edits)
        self.assertEqual(edited_code, expected_code)

    def test_mappedcode_edit_double(self):
        # input
        code = """        dog
        cat
        
        foo
        bar
        foo
        """

        edits: list[CodeEdit] = [
            CodeEdit(2, 4, "        cat"),
            CodeEdit(4, 6, "        dog"),
        ]

        mapped_code = MappedCode(code=code, feature_map=None, image=None)

        # expected
        expected_code = """        dog
        cat
        dog
        foo
        """
        edited_code = mapped_code.apply_edits(edits)
        self.assertEqual(edited_code, expected_code)

    def test_mappedcode_delete(self):
        # input
        code = """        dog
        cat
        
        foo
        bar
        foo
        """

        edits: list[CodeEdit] = [
            CodeEdit(2, 3),
        ]

        mapped_code = MappedCode(code=code, feature_map=None, image=None)

        # expected
        expected_code = """        dog
        
        foo
        bar
        foo
        """
        edited_code = mapped_code.apply_edits(edits)
        self.assertEqual(edited_code, expected_code)

    def test_mappedcode_delete_emptystring(self):
        # input
        code = """        dog
        cat
        
        foo
        bar
        foo
        """

        edits: list[CodeEdit] = [
            CodeEdit(2, 3,""),
        ]

        mapped_code = MappedCode(code=code, feature_map=None, image=None)

        # expected
        expected_code = """        dog
        
        foo
        bar
        foo
        """
        edited_code = mapped_code.apply_edits(edits)
        self.assertEqual(edited_code, expected_code)

    def test_mappedcode_insert(self):
        # input
        code = """        dog
        cat
        
        foo
        bar
        foo
        """

        edits: list[CodeEdit] = [
            CodeEdit(2, 2, "        cat"),
        ]

        mapped_code = MappedCode(code=code, feature_map=None, image=None)

        # expected
        expected_code = """        dog
        cat
        cat
        
        foo
        bar
        foo
        """
        edited_code = mapped_code.apply_edits(edits)

        print(edited_code)

        self.assertEqual(edited_code, expected_code)

    def test_mappedcode_edit_overlap(self):
        # input
        code = """        dog
        cat
        
        foo
        bar
        foo
        """

        edits: list[CodeEdit] = [
            CodeEdit(2, 5, "        cat"),
            CodeEdit(4, 4, "        dog"),
        ]

        mapped_code = MappedCode(code=code, feature_map=None, image=None)

        with self.assertRaises(ValueError):
            mapped_code.apply_edits(edits)

    def test_mappedcode_annotated(self):
        # input
        code = """        dog
        cat
        
        foo
        bar
        foo"""
        expected = """1|        dog
2|        cat
3|        
4|        foo
5|        bar
6|        foo"""

        mapped_code = MappedCode(code=code, feature_map=None, image=None)
        annotated_code = mapped_code.get_annotated()
        print(annotated_code)
        self.assertEqual(annotated_code, expected)

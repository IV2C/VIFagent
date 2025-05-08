feature_find_tool = {
    "type": "function",
    "function": {
        "name": "get_feature_location",
        "description": "Get the location of a given feature in a code. The result will be one or more parts of the code that have a high probability of being related to the feature.",
        "parameters": {
            "type": "object",
            "properties": {
                "feature_name": {
                    "type": "string",
                    "description": "Short natural language description of the visual feature to locate. In one to five words.",
                }
            },
            "required": ["feature_name"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

render_tool = {
    "type": "function",
    "function": {
        "name": "render_code",
        "description": "Renders the code and returns the image, useful to review the generated image from the code.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

modify_code_tool = {
    "type": "function",
    "function": {
        "name": "modify_code",
        "description": "modify_code(edits: List[Edit]) → code: Applies a list of textual edits and returns the modified code annotated with line numbers (for reference only). Each Edit has a start and end line index and a content. The lines in the range [start, end) are replaced by content.If start == end, content is inserted at that position, If content is an empty string, the range [start, end) is deleted.",
        "parameters": {
            "type": "object",
            "properties": {
                "edits": {
                    "type": "array",
                    "description": "List of edits to apply to the code.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "start": {
                                "type": "integer",
                                "description": "Line number of the start of the edit",
                            },
                            "end": {
                                "type": "integer",
                                "description": "Line number of the end of the edit, not selected. ",
                            },
                            "content": {
                                "type": "string",
                                "description": "New content replacing the content in the [start:end] line range.",
                            },
                        },
                        "required": ["start", "end", "content"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["edits"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

finish_customization_tool = {
    "type": "function",
    "function": {
        "name": "finish_customization",
        "description": "Tool to call when the customization is finished, i.e. when a the customization achieves the instruction wanted by the user.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

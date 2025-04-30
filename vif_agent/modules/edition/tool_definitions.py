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
        "parameters": {},
        "strict": True,
    },
}

modify_code_tool = {
    "type": "function",
    "function": {
        "name": "modify_code",
        "description": "Apply a list of edits to the code. The tool get_code must be called before this tool every time to ensure the right range is provided.",
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
                                "description": "Line number of the end of the edit.",
                            },
                            "content": {
                                "type": "string",
                                "description": "New content replacing the content in the [start:end] line range.",
                            },
                        },
                        "required": ["start", "end", "content"],
                    },
                },
            },
            "required": ["edits"],
        },
        "strict": True,
    },
}

finish_customization_tool = {
    "type": "function",
    "function": {
        "name": "finish_customization",
        "description": "Tool to call when the customization is finished, i.e. when a the customization achieves the instruction wanted by the user.",
        "parameters": {},
        "strict": True,
    },
}

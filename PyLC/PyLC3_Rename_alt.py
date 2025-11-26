import ast
def rename_variables(input_code_path: str = "generated_code_1.py",
                     blocks_module_path: str = "generated_code_0.py",
                     output_path: str = "generated_code_2.py") -> None:
    """
    Liest das in generate_python_code erzeugte Skript ein und ersetzt generische Variablennamen durch
    die in der XML definierten Namen. Schreibt das Ergebnis in output_path.
    
    Args:
        input_code_path: Pfad zur Datei mit dem generierten Python-Code ('generated_code_1.py').
        blocks_module_path: Pfad zur Datei mit den POU- und Blockdaten ('generated_code_0.py') – wird genutzt,
                            um die Variable-Mappings (input_ids) zu bestimmen.
        output_path: Pfad der zu erzeugenden Datei (z. B. 'generated_code_2.py').
    """
    def get_variable_map(pou_input_ids, pou_output_ids):
        variable_map = {}
        # Inputs
        for entry in pou_input_ids:
            expr = entry.get('Expression')
            if not expr:
                continue
            in_var = entry['InVariable'].strip()
            variable_map[f'V_{in_var}'] = expr.strip()

        # Outputs
        for entry in pou_output_ids:
            expr = entry.get('Expression')
            if not expr:
                continue
            out_var = entry['OutVariable'].strip()
            variable_map[f'V_{out_var}'] = expr.strip()
            
        return variable_map


    def replace_variable_names(code, variable_map):
        class VariableNameReplacer(ast.NodeTransformer):
            def visit_Name(self, node):
                if node.id in variable_map:
                    node.id = variable_map[node.id]
                return node

        tree = ast.parse(code)
        replacer = VariableNameReplacer()
        updated_tree = replacer.visit(tree)
        return ast.unparse(updated_tree)


    def replace_function_arguments(code, variable_map):
        class FunctionArgumentReplacer(ast.NodeTransformer):
            def visit_FunctionDef(self, node):
                for arg in node.args.args:
                    if arg.arg in variable_map:
                        arg.arg = variable_map[arg.arg]
                return self.generic_visit(node)

        tree = ast.parse(code)
        replacer = FunctionArgumentReplacer()
        updated_tree = replacer.visit(tree)
        return ast.unparse(updated_tree)


    def replace_printed_text(code, variable_map):
        class PrintTextReplacer(ast.NodeTransformer):
            def visit_Str(self, node):
                for key, value in variable_map.items():
                    if not isinstance(value, str):
                        continue
                    node.s = node.s.replace(key, value)
                return node

            # Für Python 3.10+ (Str ist Constant)
            def visit_Constant(self, node):
                if isinstance(node.value, str):
                    s = node.value
                    for key, value in variable_map.items():
                        if not isinstance(value, str):
                            continue
                        s = s.replace(key, value)
                    node.value = s
                return node

        tree = ast.parse(code)
        replacer = PrintTextReplacer()
        updated_tree = replacer.visit(tree)
        return ast.unparse(updated_tree)



    # Read the content of generated_code_1.py
    with open(input_code_path, 'r') as file:
        code_1 = file.read()

    # Read the content of generated_code_0.py
    with open(blocks_module_path, 'r') as file:
        code_0 = file.read()

    # Parse the content of generated_code_0.py to obtain the variable map
    pou_dict = {}
    exec(code_0, pou_dict)
    pou_input_ids = pou_dict['POU']['input_ids']
    variable_map = get_variable_map(pou_input_ids)

    # Replace the variable names in generated_code_1.py using the variable map
    updated_code_1 = replace_variable_names(code_1, variable_map)

    # Replace the function arguments in generated_code_1.py using the variable map
    updated_code_2 = replace_function_arguments(updated_code_1, variable_map)

    # Replace the printed text in run_cyclically() using the variable map
    updated_code_3 = replace_printed_text(updated_code_2, variable_map)

    # Write the updated code to generated_code_2.py
    with open(output_path, 'w') as file:
        file.write(updated_code_3)

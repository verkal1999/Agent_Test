import sys
import time
import inspect
import importlib.util
import textwrap


def generate_python_code(
    blocks_module_path: str = "generated_code_0.py",
    output_path: str = "generated_code_1.py"
) -> None:
    """
    Importiert das durch parse_pou_blocks erzeugte Modul (blocks_module_path) und generiert aus jedem FBD-Netzwerk
    eine Python-Funktion. Der erzeugte Code wird in output_path gespeichert.
    """
    # Modul mit POU- und Block-Definitionen laden
    spec = importlib.util.spec_from_file_location("blocks", blocks_module_path)
    blocks_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(blocks_module)

    POU = blocks_module.POU
    #blocks = [obj for name, obj in vars(blocks_module).items() if name.startswith('B')]
    blocks = [
        b for name, b in vars(blocks_module).items()
        if name.startswith('B') and b['pou_name'] == POU['pou_name']
    ]

    # Typzähler für mehrfach vorkommende Blocktypen
    type_count = {}
    for block in blocks:
        type_name = block['typeName']
        type_count[type_name] = type_count.get(type_name, 0) + 1

    # Input-Variablen-Typen aus dem POU-Dictionary
    input_var_types = POU['input_vars']

    # Mapping Expression -> Typname
    expression_to_type = {}
    for input_id, input_type in zip(POU['input_ids'], input_var_types):
        # input_type z. B. "InVar:BOOL"
        expression_to_type[input_id['Expression']] = input_type.split(':')[1]

    # Kopf des generierten Moduls
    arg_parts = []
    for input_id in POU['input_ids']:
        in_var = input_id['InVariable'].strip()
        expr = input_id['Expression']
        if expr in expression_to_type:
            arg_parts.append(f"V_{in_var}:{expression_to_type[expr]}")
        else:
            arg_parts.append(f"V_{in_var}")

    args_str = ", ".join(arg_parts)

    # Kopf des generierten Moduls
    generated_code_str = f"""
    import time
    import sys

    def {POU['pou_name']}({args_str}):
    """

    # Subfunktionen für jeden Block erzeugen
    for block in blocks:
        type_name = block['typeName']
        subfunc_name = type_name
        if type_count[type_name] > 1:
            # Instanz-Suffix anhängen
            subfunc_name += f"_{type_count[type_name]}"
            type_count[type_name] -= 1

        block_local_id = block['block_localId']
        result_var = str(int(block_local_id))

        # Funktionskopf der Blockfunktion
        generated_code_str += f"""
        def {subfunc_name}(V_{', V_'.join([var.replace(' ', '_') for var in block['inputVariables']])}):
            """

        # Zeit- / Speicherblöcke
        if type_name == 'TON':
            generated_code_str += f"""
            import time
            state = {{'Q': False, 'ET': 0, 'is_active': False, 'last_update_time': time.time()}}

            def update():
                current_time = time.time()
                elapsed_time = current_time - state['last_update_time']

                if V_{block['inputVariables'][0]}:
                    if not state['is_active']:
                        state['is_active'] = True
                        state['ET'] = 0
                        state['last_update_time'] = current_time

                    state['ET'] += elapsed_time
                    if state['ET'] >= V_20000000003:
                        state['Q'] = True
                else:
                    state['Q'] = False
                    state['ET'] = 0
                    state['is_active'] = False

            update()
            V_{block_local_id} = state['Q']
            return V_{block_local_id}
            """

        elif type_name == 'TOF':
            generated_code_str += f"""
            import time
            state = {{'Q': False, 'ET': 0, 'last_update_time': time.time()}}

            def update():
                current_time = time.time()
                elapsed_time = current_time - state['last_update_time']

                if V_{block['inputVariables'][0]}:
                    state['ET'] = 0
                elif not state['Q']:
                    state['ET'] += elapsed_time
                    if state['ET'] >= V_{block['inputVariables'][1]}:
                        state['Q'] = True

                state['last_update_time'] = current_time

            update()
            V_{block_local_id} = state['Q']
            return V_{block_local_id}
            """

        elif type_name == 'TP':
            generated_code_str += f"""
            import time
            state = {{'Q': False, 'is_active': False}}

            def update():
                if V_{block['inputVariables'][0]}:
                    if not state['is_active']:
                        state['is_active'] = True
                        state['Q'] = True
                else:
                    state['is_active'] = False
                    state['Q'] = False

            update()
            V_{block_local_id} = state['Q']
            return V_{block_local_id}
            """

        elif type_name == 'RS':
            generated_code_str += f"""
            state = {{'Q': False}}

            def update():
                if V_{block['inputVariables'][1]}:
                    state['Q'] = False
                elif V_{block['inputVariables'][0]}:
                    state['Q'] = True

            update()
            V_{block_local_id} = state['Q']
            return V_{block_local_id}
            """

        # einfache logische / arithmetische Blöcke
        else:
            input_variables = [f"V_{var.replace(' ', '_')}" for var in block['inputVariables']]

            if type_name == 'XOR':
                generated_code_str += f"V_{result_var} = {' ^ '.join(input_variables)}\n"
            elif type_name == 'AND':
                generated_code_str += f"V_{result_var} = {' and '.join(input_variables)}\n"
            elif type_name == 'OR':
                generated_code_str += f"V_{result_var} = {' or '.join(input_variables)}\n"
            elif type_name == 'NOT':
                generated_code_str += f"V_{result_var} = {' not '.join(input_variables)}\n"
            elif type_name == 'R_TRIG':
                generated_code_str += f"V_{result_var} = {''.join(input_variables)}\n"
            elif type_name == 'ADD':
                generated_code_str += f"V_{result_var} = {' + '.join(input_variables)}\n"
            elif type_name == 'LT':
                generated_code_str += f"V_{result_var} = {' < '.join(input_variables)}\n"
            elif type_name == 'GE':
                generated_code_str += f"V_{result_var} = {' >= '.join(input_variables)}\n"
            elif type_name == 'LE':
                generated_code_str += f"V_{result_var} = {' <= '.join(input_variables)}\n"

            # return MUSS in der Blockfunktion bleiben → 12 Spaces
            generated_code_str += f"            return V_{result_var}\n\n"

        # Aufruf der Blockfunktion im POU-Body → 8 Spaces
        generated_code_str += f"""
        V_{block_local_id} = {subfunc_name}(V_{', V_'.join([var.replace(' ', '_') for var in block['inputVariables']])})
        """

    # Zuordnung der Ausgänge im POU-Body
    for out_id in POU['output_ids']:
        out_var = out_id['OutVariable'].strip()
        generated_code_str += (
            f"\n        V_{out_var} = V_{out_var[:-1]}{int(out_var[-1]) - 1}"
        )
        generated_code_str += (
            f"\n        print('Value of V_{out_var}:', V_{out_var})\n"
        )

    # f-String-Ausgabe vorbereiten
    output_variables = []
    for out_id in POU['output_ids']:
        out_var = out_id['OutVariable'].strip()
        output_variables.append(f"{out_var}:{{V_{out_var}}}")

    return_expr = " ".join(output_variables)
    # return im POU-Body (8 Spaces)
    generated_code_str += f"\n        return f\"{return_expr}\"\n"

    # Typnamen in Python-Typen umschreiben
    generated_code_str = (
        generated_code_str
        .replace('BOOL', 'bool')
        .replace('TIME', 'int')
        .replace('INT', 'int')
        .replace('STRING', 'str')
        .replace('CHAR', 'str')
        .replace('WCHAR', 'str')
        .replace('WSTRING', 'str')
    )

    # Einrückung begradigen
    generated_code_str = textwrap.dedent(generated_code_str)
    print(generated_code_str)

    # In Datei schreiben
    with open(output_path, 'w') as file:
        file.write(generated_code_str)

        # External loop module (Top-Level)
        file.write(textwrap.dedent('''
    # External loop module
    def run_cyclically():
        # Add str_to_bool and str_to_int functions inside run_cyclically
        def str_to_bool(s):
            return s.lower() in ('true', 't', '1')

        def str_to_int(s):
            try:
                return int(s)
            except ValueError:
                print("Invalid input. Please enter a valid integer.")
                return None

        for i in range(5):
            print(f"Iteration {i + 1}")
    '''))

        # Inputvariablen für run_cyclically vorbereiten
        input_variable_names = [
            f"V_{input_id['InVariable'].strip()}" for input_id in POU['input_ids']
        ]
        input_variable_types = [
            expression_to_type.get(input_id['Expression'], 'TIME')
            for input_id in POU['input_ids']
        ]
        input_variable_types = [
            t.replace('BOOL', 'bool').replace('TIME', 'int').replace('INT', 'int')
            for t in input_variable_types
        ]

        # Eingabeprompts im for-Loop
        for var_name, var_type in zip(input_variable_names, input_variable_types):
            if var_type == 'bool':
                file.write(
                    f"        {var_name} = str_to_bool(input(f\"Enter value for {var_name} ({var_type}): \"))\n"
                )
            elif var_type == 'int':
                if var_name == 'V_T#200ms':
                    file.write(f"        {var_name} = 200\n")
                else:
                    file.write(
                        f"        {var_name} = str_to_int(input(f\"Enter value for {var_name} ({var_type}): \"))\n"
                    )
            else:
                file.write(
                    f"        {var_name} = {var_type}(input(f\"Enter value for {var_name} ({var_type}): \"))\n"
                )

        function_name = POU['pou_name']
        file.write(f"        result = {function_name}(")
        file.write(", ".join(input_variable_names))
        file.write(")\n")
        file.write("        print('Result:', result)\n")
        file.write("        time.sleep(3)\n")
        file.write("# Run the cyclic execution\n")
        file.write("run_cyclically()\n")

    print("Generated code has been written to ", output_path, ".")

import os
import sys

# This script assumes that hf_llama.py was run 
# in select_skill/ select_skill_dungeons/ and select_skill_mines/

def extract_mines_code(mines_path):
    with open(mines_path, 'r') as file:
        lines = file.readlines()
    
    extracted_code = []
    for line in lines:
        if "# starting conditions" in line.strip().lower():
            break
        extracted_code.append(line)

    return ''.join(extracted_code)

def extract_dungeons_code(dungeons_path):
    """
    Extracts the content of the dungeons execution.py file from the function definition
    'def reach_dungeons_of_doom' up until the '# Starting conditions' comment.
    """
    with open(dungeons_path, 'r') as file:
        lines = file.readlines()

    start_collecting = False
    extracted_code = []
    
    for line in lines:
        if "def reach_dungeons_of_doom" in line:
            start_collecting = True
        if start_collecting:
            if "# starting conditions" in line.strip().lower():
                break
            extracted_code.append(line)

    return ''.join(extracted_code)

def combine_code(mines_code, dungeons_code):
    """
    Combines the extracted code from both files.
    """
    return mines_code + dungeons_code

def save_combined_code(combined_code, output_path):
    """
    Saves the combined code to the specified output path.
    """
    with open(output_path, 'w') as file:
        file.write(combined_code)

def main():
    seed = 42 if len(sys.argv) != 2 else sys.argv[1]

    mines_path = f'reach_gnomish_mines/seed{seed}/execution.py'
    dungeons_path = f'reach_dungeons_of_doom/seed{seed}/execution.py'
    os.makedirs(f'seed{seed}', exist_ok=True)
    output_path = f'seed{seed}/execution.py'
    
    mines_code = extract_mines_code(mines_path)
    dungeons_code = extract_dungeons_code(dungeons_path)
    combined_code = combine_code(mines_code, dungeons_code)

    with open('perform_task.txt', 'r') as file:
        lines = file.readlines()
        task = "".join(lines)
    combined_code += task

    with open('termination.txt', 'r') as file:
        lines = file.readlines()
        term = "".join(lines)
    combined_code += term

    with open('precondition.txt', 'r') as file:
        lines = file.readlines()
        precon = "".join(lines)
    combined_code += precon

    with open('initial_values.txt', 'r') as file:
        lines = file.readlines()
        initval = "".join(lines)
    combined_code += initval

    with open(output_path, 'w') as file:
        file.write(combined_code)
    with open(f'seed{seed}/__init__.py', 'w') as file:file.write('')

    print(f"Combined code has been saved to {output_path}")



if __name__ == "__main__":
    main()

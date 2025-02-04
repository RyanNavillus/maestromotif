import os 
import shutil
import subprocess
import sys
from huggingface_hub import InferenceClient


client = InferenceClient(
    token="your_token",
)

temperature = 0.2
messages = [{"role": "system", "content": "You are a helpful assistant."},]
model = "meta-llama/Meta-Llama-3.1-405B-Instruct"
max_tokens = 5000

# initial message
with open("initial.txt", "r") as f:
    initial = f.read()
messages.append({"role": "user", "content": initial})

completion = client.chat.completions.create(
    model=model,
    messages=messages,
    temperature=temperature,
    max_tokens=max_tokens,
)
output = completion.choices[0].message.content
messages.append({"role": "assistant", "content": output})

try:
    code = output.split('```python')[1].split('```')[0]
    if 'for turn in range' not in code:
        with open('unit_test.py', 'r') as f:unit_test = f.readlines()
        utest = ''.join(unit_test)
        code += utest
    with open("execution.py", 'w') as f: f.write(code)
    result = subprocess.run(['python', "execution.py"], capture_output=True, text=True)
except:
    print("\n\n\nthere is a problem...")
    input()

print(result.stdout)

# code checking
with open("code_check.txt", 'r') as f:
    code_check = f.read()
code_check = code_check.format(result.stdout)
messages.append({"role": "user", "content": code_check})
completion = client.chat.completions.create(
    model=model,
    messages=messages,
    temperature=temperature,
    max_tokens=max_tokens,
)
output = completion.choices[0].message.content
print(output)
messages.append({"role": "assistant", "content": output})
input()

iterations = 1
while 'yes' not in output.lower() and iterations < 5:
    print("\nimprovement iteration ", iterations, "\n")

    # simulate and rewrite
    with open("simulate_rewrite.txt", 'r') as f:
        simulate_rewrite = f.read()
    messages.append({"role": "user", "content": simulate_rewrite})
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    output = completion.choices[0].message.content
    messages.append({"role": "assistant", "content": output})

    try:
        code = output.split('```python')[1].split('```')[0]
        with open("execution.py", 'w') as f: f.write(code)
        result = subprocess.run(
            ['python', "execution.py"], capture_output=True, text=True)
    except:
        print("\n\n\nthere is a problem...")
        input()
    
    print(result.stdout)

    # output checking
    with open("code_check.txt", 'r') as f:
        code_check = f.read()
    code_check = code_check.format(result.stdout)   
    messages.append({"role": "user", "content": code_check})
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    output = completion.choices[0].message.content
    print(output)
    messages.append({"role": "assistant", "content": output})
    input()
    
    iterations += 1


### Saving important info

seed = 42 if len(sys.argv) != 2 else sys.argv[1]

source_dir = os.getcwd()
destination_dir = os.path.join(source_dir, f'seed{seed}')
os.makedirs(destination_dir, exist_ok=True)

item_path = os.path.join(source_dir, 'execution.py')

# Check if the item is a file (not a directory)
if os.path.isfile(item_path):
    shutil.copy(item_path, destination_dir)
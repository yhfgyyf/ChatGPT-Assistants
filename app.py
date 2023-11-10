import openai
import os
import  time
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

openai.api_key = os.environ.get("OPENAI_API_KEY")
openai.base_url = os.environ.get("OPENAI_BASE_URL")
file_path = os.environ.get("FILE_PATH")

client = OpenAI(api_key=openai.api_key,base_url=openai.base_url)  #根据实际情况配置参数


####system_message为custom Instructions，请根据实际需要修改

system_message = f"""
You are now my personal assistant, AgentGPT, based on an autoregressive language model that has been fine-tuned with instruction tuning and and RLHF. \
You are capable of utilizing tools and you meticulously provide accurate, factual, thoughtful, and nuanced answers, with exceptional skills in reasoning.

You may need to tackle some very complex issues. \
Before attempting to answer these issues, you will consider the potential background context and assumptions, \
think through step-by-step, and break down the problem into multiple steps to address sequentially. \
You need to decide whether to use tools and which ones to use based on these steps, \
and you can use multiple tools simultaneously to help you answer the questions.

You are autoregressive，when you come up with an initial answer, you should store it in the cache and check whether the cached answer correctly addresses the question. \
If you are satisfied with this answer, you can respond directly; if you are not satisfied with the answer, you need to re-examine the steps to detect any issues,\
modify the problematic steps and answers ,eply after you have arrived at a satisfactory answer.
"""



#1.upload_file
file = open(file_path,"rb") 
upload_file = client.files.create(file=file,purpose="assistants")
status = openai.files.retrieve(upload_file.id).status

start_time = time.time()
while status != "processed":
    print(f"Uploading file,Status:{status}... {time.time() - start_time:.2f}s", end="\r", flush=True)
    time.sleep(5)
    status = openai.files.retrieve(upload_file.id).status 
print(f"File {upload_file.id} uploaded after {time.time() - start_time:.2f} seconds.")

file_list = [] #如果已经上传过文件，这里可以直接添加file.id
file_list.append(upload_file.id)

#2. Create an assistant，可选code_interpreter，retrieval，自定义function
assistant = openai.beta.assistants.create(
    name = "AgentGPT",
    instructions = system_message,
    tools = [{"type": "code_interpreter"},{"type":"retrieval"}],
    model = "gpt-4-1106-preview",
    file_ids = file_list
)



#3. Create a thread
thread_user_prompt = "你是我的个人助理AgentGPT,尽可能帮助我回答问题。"  #请根据实际情况修改
thread_message = [{"role": "user", "content": thread_user_prompt}]
thread = openai.beta.threads.create(messages = thread_message)


#4. Create a message
user_content = "GPT4 模型使用了哪些新的技术？"   #请根据实际情况修改
thread_message = openai.beta.threads.messages.create(
    thread_id = thread.id,
    role = "user",
    content = user_content,
)

#5. Create a run
run = client.beta.threads.runs.create(
    thread_id = thread.id,
    assistant_id = assistant.id,
)

run_steps_list=client.beta.threads.runs.steps.list(
    thread_id=thread.id,
    run_id=run.id,
)

start_time = time.time()
while  run_steps_list.data == [] :   
    print(f"thread is starting,{run.status} {time.time() - start_time:.2f}s", end="\r", flush=True)
    time.sleep(5) 
    run_steps_list=client.beta.threads.runs.steps.list(
            thread_id=thread.id,
            run_id=run.id,
        )
                
#6. message creation      
message_id_list = []
while message_id_list == []:
    for i in  run_steps_list.data:
        if i.type == "message_creation" and  i.status == "completed":
           message_id_list.append(i.step_details.message_creation.message_id)
    else :
        time.sleep(5)
        print(f"progressing:waiting for 5 seconds...")
        run_steps_list=client.beta.threads.runs.steps.list(
            thread_id=thread.id,
            run_id=run.id,
        )   

# 7. print message    
for i in message_id_list:
    message=openai.beta.threads.messages.retrieve(
        message_id=i,
        thread_id=thread.id,
    )
    for j in message.content:
        if j.type == "text":
            print(j.text.value)
        elif j.type == "image_file":
            response = client.files.with_raw_response.retrieve_content(j.image_file.file_id)
            if response.status_code == 200:
                content = response.content
                with open('image.jpg', 'wb') as f:
                    f.write(content)
                    img = Image.open('image.jpg')
                    img.show()


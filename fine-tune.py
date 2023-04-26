import os
import openai
import subprocess
import pandas as pd
#設定自己的金鑰
openai.api_key_path = "D:\openaikey.txt"

#1.把chatgpt的回覆存到csv檔裡
#把可以調換的關鍵詞都先列舉出來
l_country = ['台灣', '美國', '日本', '英國', '澳洲']
l_direction = ['東', '西', '南', '北']
#把剛剛的關鍵詞塞進自己想得到的問題當中
f_prompt = "請問位於{country}最{direction}的城市是甚麼，並把它地理位置和人口數用中文回答"

f_sub_prompt = " {country},{direction}"

df = pd.DataFrame()
#把所以可能的關鍵詞組合都問chatgpt一次
for country in l_country:
    for direction in l_direction:
            for i in range(3):  ## 同個問題跑三遍得到三種答案
                #把完整附關鍵詞之問題存在prompt並顯示當前是問哪組關鍵詞組合
                prompt = f_prompt.format(country=country, direction=direction)
                sub_prompt = f_sub_prompt.format(country=country, direction=direction)
                print(sub_prompt)
                #與作業1相同
                response = openai.Completion.create(
                    model="text-davinci-003",
                    prompt=prompt,
                    temperature=1,
                    max_tokens=100,
                    top_p=0.75,
                    frequency_penalty=0,
                    presence_penalty=0
                )
                finish_reason = response['choices'][0]['finish_reason']#此存放關鍵詞
                response_txt = response['choices'][0]['text']#此存放回答
                # 先new一個空間把關鍵詞、問題、回答等內容記錄下來(形式 欄位名稱:內容)
                new_row = {
                    'country': country,
                    'direction': direction,
                    'prompt': prompt,
                    'sub_prompt': sub_prompt,
                    'response_txt': response_txt,
                    'finish_reason': finish_reason}
                # 再把剛剛紀錄的關鍵詞、問題、回答等內容記寫進csv檔裡
                new_row = pd.DataFrame([new_row])
                df = pd.concat([df, new_row], axis=0, ignore_index=True)

df.to_csv("D:\out_openai_completion.csv")

#2.把csv檔裡資料調整成自己想要的
df = pd.read_csv("D:\out_openai_completion.csv")
#把out_openai_completion.csv的資料去掉標點符號
prepared_data = df.loc[:,['sub_prompt','response_txt']]
#接著重新命名參數
prepared_data.rename(columns={'sub_prompt':'prompt', 'response_txt':'completion'}, inplace=True)
#在後加載到prepared_data.csv
prepared_data.to_csv('D:\prepared_data.csv',index=False)

#輸入文件prepared_data.csv檢查數據是否正確，並生成一個名為prepared_data_prepared.jsonl的文件
subprocess.run('openai tools fine_tunes.prepare_data --file D:\prepared_data.csv --quiet'.split())
#給出之前創建的 JSONL 文件的名稱。然後依照自己所期望的選擇來微調模型
subprocess.run('openai api fine_tunes.create --training_file prepared_data_prepared.jsonl --model davinci --suffix "SuperHero"'.split())

response = openai.Completion.create(
        engine="自己的訓練模型名稱", #選擇要使用openai中的哪一種引擎模型
        prompt="印度,東 ->\n",
        max_tokens=30,             #要chatGPT回復的字串長度
        #temperature和top_p式調整回覆內容多元性設定，1越多元，0月制式
        temperature=1,
        top_p=0.75,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["END"]
)
#把json格式檔的回覆存取下來
completed_text = response["choices"][0]["text"]

try:#fread
    with open("D:\\ansofGPT.txt") as file: #檢查檔案是否存在
        print(file.read())
except FileNotFoundError:
    print("name error")

with open("D:\\ansofGPT.txt","a") as file:
     file.write(completed_text)         #把存取下來的回答寫進file裡

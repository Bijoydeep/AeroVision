import openai
openai.api_key = "your-api-key"
def explain_trend(data):
    text_data = data.tail(30).to_csv()
    prompt = f"""You are an expert in environmental science.
Analyze the following 30-day air pollution data and summarize the trends and any anomalies.
{str(text_data)}"""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message["content"]

import re

pattern = re.compile(r"<\|(\d+\.\d+)\|>([^<]+)<\|(\d+\.\d+)\|>")
input_string = (
    "<|0.00|> and the 16 years that we've been married.<|2.00|><|2.00|> Have you one"
    " time told off that you liked him?<|5.00|><|5.00|> Not in those exact"
    " words.<|7.00|><|7.00|> No.<|8.00|><|8.00|> No.<|9.00|><|9.00|> Not in any words,"
    " Dad.<|10.00|><|10.00|> He said that makes me feel.<|12.00|><|13.00|> You've never"
    " told your son that you love him.<|16.00|>"
)


# Find all matches in the text
matches = pattern.findall(input_string)

# Convert matches to a list of tuples (start_timestamp, end_timestamp, text)
result = {}
result["text"] = ""
result["chunks"] = []
for i in range(len(matches)):
    transcription = matches[i][1].strip()
    result["text"] += transcription
    segment = {}
    segment["timestamp"] = (float(matches[i][0]), float(matches[i][2]))
    segment["text"] = transcription
    result["chunks"].append(segment)
print(result)

import re

def parse_string_into_result(input_string, language):
    pattern = re.compile(r'<\|(\d+\.\d+)\|>([^<]+)<\|(\d+\.\d+)\|>')

    # Find segments in the text
    matches = pattern.findall(input_string)

    # Convert segments to dictionary
    result = {}
    result['language'] = language
    result['text'] =''
    result['chunks']=[0]*len(matches)
    for idx, match in enumerate(matches):
        result['text'] += match[1]
        segment = {}
        segment['timestamp'] = (float(match[0]), float(match[2]))
        segment['text'] = match[1]
        result['chunks'][idx] = segment
    return result

input_string = "<|0.00|> and the 16 years that we've been " \
          "married.<|2.00|><|2.00|> Have you one time told off that you liked him?<|5.00|><|5.00|> Not in those exact words.<|7.00|><|7.00|> No.<|8.00|><|8.00|> No.<|9.00|><|9.00|> Not in any words, Dad.<|10.00|><|10.00|> He said that makes me feel.<|12.00|><|13.00|> You've never told your son that you love him.<|16.00|>"

result = parse_string_into_result(input_string, "en")
print(result)
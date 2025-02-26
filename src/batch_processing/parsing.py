import re
import os
import pandas as pd


def parse_string_into_result(input_string, language):
    pattern = re.compile(r"<\|(\d+\.\d+)\|>([^<]+)<\|(\d+\.\d+)\|>")

    # Find segments in the text
    matches = pattern.findall(input_string)

    # Convert segments to dictionary
    result = {}
    result["language"] = language
    if not matches:
        # if matches are empty, no segments
        # Remove random timestamps
        result["text"] = re.sub(r'<\|.*?\|> |!\s', '', input_string).strip()
        result["chunks"] = [{"timestamp": (0, 30), "text": input_string}]
    else:
        result["text"] = ""
        result["chunks"] = [0] * len(matches)
        for idx, match in enumerate(matches):
            result["text"] += match[1]
            segment = {}
            segment["timestamp"] = (float(match[0]), float(match[2]))
            segment["text"] = match[1]
            result["chunks"][idx] = segment
    return result


def parse_results_to_csv(result, output_dir, output_name):
    # Translate text into Whisper
    segment_len = len(result["chunks"])
    transcribe_df = pd.DataFrame()
    start_list = [0] * segment_len
    end_list = [0] * segment_len
    text_list = [0] * segment_len

    for idx, segment in enumerate(result["chunks"]):
        start_list[idx], end_list[idx], text_list[idx] = (
            segment["timestamp"][0],
            segment["timestamp"][1],
            segment["text"],
        )

    transcribe_df["start"], transcribe_df["end"], transcribe_df["text"] = (
        start_list,
        end_list,
        text_list,
    )
    transcribe_df["file_name"] = output_name
    transcribe_df["speaker"] = ""

    # Remove leading and trailing whitespaces from whisper outputs
    transcribe_df["text"] = transcribe_df["text"].apply(lambda x: x.strip())

    transcribe_df.to_csv(
        os.path.join(output_dir, "{}.csv".format(output_name)), index=False
    )
    return transcribe_df


input_string = (
    "<|0.00|> an ofd the 16 years that we've been married.<|2.00|><|2.00|> Have you one"
    " time toldf that you liked him?<|5.00|><|5.00|> Not in those exact"
    " words.<|7.00|><|7.00|> No.<|8.00|><|8.00|> No.<|9.00|><|9.00|> Not in any words,"
    " Dad.<|10.00|><|10.00|> He said that makes me feel.<|12.00|><|13.00|> You've never"
    " told your son that you love him.<|16.00|>"
)

input_string=" ! Good morning. I now call this meeting of the Abilene City Council order is 830 a.m. I'm going to introduce here in a moment but I'm going to ask Councilman McAllister if he would lead us in our invocation. Let's have a word. Dearly Father, we are very blessed to be here today. We thank you for this opportunity and it is as we know an opportunity to serve you in so many different ways. We pray this morning that we"
result = parse_string_into_result(input_string, "en")
print(result)


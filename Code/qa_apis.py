import torch

def get_ans_from_context(model, tokenizer, context, question):

    inputs = tokenizer.encode_plus(question, context, return_tensors="pt")

    answer_start_scores, answer_end_scores = model(**inputs)
    answer_start = torch.argmax(answer_start_scores)

    answer_end = torch.argmax(answer_end_scores) + 1
    span_score = torch.max(answer_start_scores) + torch.max(answer_end_scores)
    return tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end])), span_score
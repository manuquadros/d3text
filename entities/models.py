import transformers


def biolinkbert_large() -> (
    tuple[
        transformers.models.bert.tokenization_bert_fast.BertTokenizerFast,
        transformers.models.bert.modeling_bert.BertModel,
    ]
):
    return (
        transformers.AutoTokenizer.from_pretrained("michiyasunaga/BioLinkBERT-large"),
        transformers.AutoModel.from_pretrained("michiyasunaga/BioLinkBERT-large"),
    )

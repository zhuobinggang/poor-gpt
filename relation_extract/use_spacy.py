import spacy
nlp = spacy.load('en_core_web_sm')

text = 'Once upon a time in a certain country there lived a king whose palace was surrounded by a spacious garden. But, though the gardeners were many and the soil was good, this garden yielded neither flowers nor fruits, not even grass or shady trees. The King was in despair about it, when a wise old man said to him: “Your gardeners do not understand their business: but what can you expect of men whose fathers were cobblers and carpenters? How should they have learned to cultivate your garden?” '

doc = nlp(text)
sents = list(doc.sent)

def get_all_entity_from_sent(text):
    res = []
    for tok in nlp(text):
        if tok.pos_ in ['PROPN', 'NOUN']:
            res.append(tok.text)
    return res

def get_all_important_entity_from_sent(text):
    res = []
    for tok in nlp(text):
        if tok.pos_ in ['PROPN']:
            res.append(tok.text)
    return res


# From Net
def get_entities(sent):
  ## chunk 1
  ent1 = ""
  ent2 = ""
  prv_tok_dep = ""    # dependency tag of previous token in the sentence
  prv_tok_text = ""   # previous token in the sentence
  prefix = ""
  modifier = ""
  #############################################################
  for tok in nlp(sent):
    ## chunk 2
    # if token is a punctuation mark then move on to the next token
    if tok.dep_ != "punct":
      # check: token is a compound word or not
      if tok.dep_ == "compound":
        prefix = tok.text
        # if the previous word was also a 'compound' then add the current word to it
        if prv_tok_dep == "compound":
          prefix = prv_tok_text + " "+ tok.text
      # check: token is a modifier or not
      if tok.dep_.endswith("mod") == True:
        modifier = tok.text
        # if the previous word was also a 'compound' then add the current word to it
        if prv_tok_dep == "compound":
          modifier = prv_tok_text + " "+ tok.text
      ## chunk 3
      if tok.dep_.find("subj") == True:
        ent1 = modifier +" "+ prefix + " "+ tok.text
        prefix = ""
        modifier = ""
        prv_tok_dep = ""
        prv_tok_text = ""
      ## chunk 4
      if tok.dep_.find("obj") == True:
        ent2 = modifier +" "+ prefix +" "+ tok.text
      ## chunk 5
      # update variables
      prv_tok_dep = tok.dep_
      prv_tok_text = tok.text
  #############################################################
  return [ent1.strip(), ent2.strip()]


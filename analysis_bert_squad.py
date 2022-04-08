# -*- coding: utf-8 -*-
"""analysis_BERT_squad.ipynb
"""

!pip install transformers

from transformers import pipeline

qa_pipeline = pipeline(
    "question-answering",
    model="deepset/bert-base-uncased-squad2",
    tokenizer="deepset/bert-base-uncased-squad2"
)

"""###Failed even though it was trained on multiple sentence reasoning
Although it performs pretty well on complex questions. A change in context, and the model doesn't realize it. The key idea behind this is that it was pretrained on Wikipedia datasets where each sentence is coherent with the previous one. 
"""

qa_pipeline({
    'context': "My girlfriend's name is Sarah and she lives in London. I am very passionate about cricket and my favorite team is RCB.",
    'question': "What is Sarah's boyfriend's favorite team?"

})

qa_pipeline({
    'context': "Her name is Sarah and she lives in London. I am passionate about cricket and my favorite team is RCB.",
    'question': "Where do I live?"

})

qa_pipeline({
    'context': "Her name is Sarah and she lives in London. I am very passionate about cricket and my favorite team is RCB.",
    'question': "Which is Sarah's favorite team?"

})

qa_pipeline({
    'context': "Her name is Sarah and she lives in London. I am very passionate about cricket and my favorite cricket team is RCB.",
    'question': "Which is my favorite football team?"

})

"""
**Fix it**
"""

qa_pipeline({
    'context': 'The English name "Normans" comes from the French words Normans/Normanz, plural of Normant, modern French normand, which is itself borrowed from Old Low Franconian Nortmann "Northman" or directly from Old Norse Norðmaðr, Latinized variously as Nortmannus, Normannus, or Nordmannus (recorded in Medieval Latin, 9th century) to mean "Norseman, Viking".',
    'question': "What is the original meaning of the word Norman?"

})

"""### Failure multiple answers

"""

qa_pipeline({
    'context': '''Queen are a British rock band formed in London in 1970. Their classic line-up was Freddie Mercury (lead vocals, piano), 
            Brian May (guitar, vocals), Roger Taylor (drums, vocals) and John Deacon (bass). Their earliest works were influenced 
            by progressive rock, hard rock and heavy metal, but the band gradually ventured into more conventional and radio-friendly 
            works by incorporating further styles, such as arena rock and pop rock.''',
    'question': "Who were the basic members of Queen band?"

})

"""On Dev Set, 
1.   Percentage of correct answers by date/noun/verb, etc.
2.   Syntactic divergence
"""


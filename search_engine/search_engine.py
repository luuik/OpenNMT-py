import nltk
import os, os.path
from whoosh import index
from whoosh.fields import Schema, ID, TEXT
from whoosh.qparser import QueryParser
from whoosh import qparser
from whoosh import scoring

schema = Schema(original=ID(unique=True, stored=True), content=TEXT, translation=ID(unique=False, stored=True))
if not os.path.exists("indexdir"):
    os.mkdir("indexdir")
    ix = index.create_in( "indexdir" , schema )
else: ix = index.open_dir("indexdir")

def add_sentences( file ):
    writer = ix.writer()
    for line in open(file, encoding="utf-8"):
        line = line.split("\t")
        writer.update_document(original=line[0],content=line[0],translation=line[1])
    writer.commit()

def index_corpus( limit=-1 ):
    writer = ix.writer()
    data1 = open("JRC-Acquis.en-es.en", encoding="utf-8").readlines()
    data2 = open("JRC-Acquis.en-es.es", encoding="utf-8").readlines()
    for pos, s in enumerate(data1):
        writer.update_document(original=s.lower(),content=s.lower(),translation=data2[pos].lower())
        if (pos != 0) and (pos % 10000 == 0): print(pos)
        if (limit > 0 and pos == limit): break
    writer.commit()

def retrieve_sentences( sentence , limit1=5 , limit2=5 ):
    searcher = ix.searcher()
    qp = QueryParser("content", schema=schema, group=qparser.OrGroup)
    op = qparser.OperatorsPlugin(And="&", Or="\\|", AndNot="&!", AndMaybe="&~", Not="\\-")
    qp.replace_plugin(op)
    q = qp.parse(sentence)
    results = ix.searcher(weighting=scoring.BM25F(B=0.75, content_B=1.0, K1=1.5)).search(q, limit=limit1)
    results = [ ( i["original"] , i["translation"] ) for i in results[0:limit1] ]
    scores = [ nltk.translate.bleu_score.sentence_bleu([sentence], h[0], weights = (0.5, 0.5)) for h in results ]    
    results = [ (y,x[0],x[1]) for y,x in sorted(zip(scores,results), reverse=True) ]
    return results[0:limit2]

index_corpus()
results = retrieve_sentences( sentence=u"looking for a job" , limit1=500 , limit2=5)
print(results)

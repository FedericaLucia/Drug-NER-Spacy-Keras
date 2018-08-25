############################################  NOTE  ########################################################
#
#           Creates NER training data in Spacy format from JSON downloaded from Dataturks.
#
#           Outputs the Spacy training data which can be used for Spacy training.
#
############################################################################################################
import json
import random
import logging
import sys
from collections import defaultdict
from pathlib import Path
from spacy.gold import GoldParse
from spacy.scorer import Scorer
from sklearn.metrics import accuracy_score
from spacy.util import minibatch, compounding
from spacy.util import decaying


def convert_dataturks_to_spacy(dataturks_JSON_FilePath):
    try:
        training_data = []
        lines=[]
        with open(dataturks_JSON_FilePath, 'r') as f:
            lines = f.readlines()

        for line in lines:
            data = json.loads(line)
            text = data['content']
            entities = []
            for annotation in data['annotation']:
                #only a single point in text annotation.
                point = annotation['points'][0]
                labels = annotation['label']
                # handle both list of labels or a single label.
                if not isinstance(labels, list):
                    labels = [labels]

                for label in labels:
                    #dataturks indices are both inclusive [start, end] but spacy is not [start, end)
                    entities.append((point['start'], point['end'] + 1 ,label))


            training_data.append((text, {"entities" : entities}))

        return training_data
    except Exception as e:
        logging.exception("Unable to process " + dataturks_JSON_FilePath + "\n" + "error = " + str(e))
        return None

import spacy
################### Train Spacy NER.###########
def train_spacy():
    drug_model=("Drug model.", "option", "nm", str),
    TRAIN_DATA = convert_dataturks_to_spacy("/Users/federicavinella/Desktop/BNF_traindata_dataturks.JSON")
    nlp = spacy.load('en')  # create blank Language class
    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
       

    # add labels
    for _, annotations in TRAIN_DATA:
         for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        max_batch_sizes = {'tagger': 32, 'parser': 16, 'ner': 16, 'textcat': 64}
        max_batch_size = max_batch_sizes['ner']
        
    if len(TRAIN_DATA) < 1000:
        max_batch_size /= 2
    if len(TRAIN_DATA) < 500:
        max_batch_size /= 2
        for itn in range(50):
            print("Statring iteration " + str(itn))
            random.shuffle(TRAIN_DATA)
            losses = {}
            batch_size = compounding(1., 8., 1.05)
            batches = minibatch(TRAIN_DATA, size=batch_size)
            for batches, annotations in TRAIN_DATA:
                nlp.update(
                    [batches],  # batch of texts
                    [annotations],  # batch of annotations
                    drop=0.1,  # dropout - make it harder to memorise data
                    sgd=optimizer,  # callable to update weights                   
                    losses=losses)           
            print(losses)
 


    # save model to output directory
    output_dir = Path("/Users/federicavinella/Desktop/BNF_POS")
    if output_dir is not None:
        output_dir = Path("/Users/federicavinella/Desktop/BNF_POS")
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.meta['name'] = drug_model  # rename model
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)



    nlp = spacy.load(output_dir)
    text = u' Ketamine is clinically compatible with the commonly used general and local anesthetic agents when an adequate respiratory exchange is maintained.Ketoconazole is a potent inhibitor of the cytochrome P450 3A4 enzyme system.Coadministration of NIZORAL  Tablets and drugs primarily metabolized by the cytochrome P450 3A4 enzyme system may result in increased plasma concentrations of the drugs that could increase or prolong both therapeutic and adverse effects.Therefore, unless otherwise specified, appropriate dosage adjustments may be necessary.The following drug interactions have been identified involving NIZORAL Tablets and other drugs metabolized by the cytochrome P450 3A4 enzyme system: Ketoconazole tablets inhibit the metabolism of terfenadine, resulting in an increased plasma concentration of terfenadine and a delay in the elimination of its acid metabolite.Diuretic:  Hydrochlorothiazide, given concomitantly with ketoprofen, produces a reduction in urinary potassium and chloride excretion compared to hydrochlorothiazide alone.Usually, this has been observed in patients with a history of diabetes mellitus or evidence of glucose intolerance prior to administration of CAMPTOSAR.'
    doc = nlp(text)
    print ('--- Predicted Labels ---')
    for ent in doc.ents:
        
        print(ent.label_, ent.text)

#    print ('--- Tokens ---')
#    for tok in doc:
#        print (tok.i, tok)   
#    print ('')

    with nlp.disable_pipes('ner'):
        doc = nlp(text)

    (beams, somethingelse) = nlp.entity.beam_parse([ doc ], beam_width = 8, beam_density = 0.0001)

    entity_scores = defaultdict(float)
    for beam in beams:
        for score, ents in nlp.entity.moves.get_beam_parses(beam):
            for start, end, label in ents:
                entity_scores[(start, end, label)] += score

    print ('--- Entities and scores (detected with beam search) ---')
    for key in entity_scores:
        start, end, label = key
        print ('%d to %d: %s (%f)' % ( start, end - 1, label, entity_scores[key]))
 
train_spacy()



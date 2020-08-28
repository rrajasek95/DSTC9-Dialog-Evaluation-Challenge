import argparse
import json
import os
import random
from collections import defaultdict
import pandas as pd


def analyze_da_information_for_split(da_mapping):

    for da, segments in da_mapping.items():
        print("Label:", da)
        print("Frequency", len(segments))
        print("Examples:")
        sample_segments = random.sample(segments, min(20, len(segments)))

        for segment in sample_segments:
            print(segment['text'])
        print("\n")


def analyze_split(split_data):
    mezza_da_segment_mapping = defaultdict(list)
    switchboard_da_segment_mapping = defaultdict(list)

    for conv_id, conv_data in split_data.items():
        turns = conv_data["content"]

        for turn in turns:
            segments = turn['segments']
            mezza_das = turn['mezza_da']
            swbd_das = turn['swbd_da_v3']

            for (segment, mezza_da, swbd_da) in zip(segments, mezza_das, swbd_das):
                mezza_da_segment_mapping[mezza_da['da']].append(segment)
                switchboard_da_segment_mapping[swbd_da['label']].append(segment)
    print("Mezza DA:")
    analyze_da_information_for_split(da_mapping=mezza_da_segment_mapping)
    print("Switchboard DA:")
    analyze_da_information_for_split(da_mapping=switchboard_da_segment_mapping)

def analyze_split_ath(split_data):
    athena_da_segment_mapping = defaultdict(list)

    for conv_id, conv_data in split_data.items():
        turns = conv_data["content"]

        for turn in turns:
            segments = turn['segments']
            athena_das = turn['athena_das']

            for (segment, athena_da) in zip(segments, athena_das):
                athena_da_segment_mapping[athena_da].append(segment)

    print("Athena DA:")
    analyze_da_information_for_split(athena_da_segment_mapping)

def analyze_length_bin_distribution(split_data):

    bin_tokens = defaultdict(list)
    for conv_id, conv_data in split_data.items():
        turns = conv_data['content']

        for turn in turns:
            segments = turn['segments']

            for segment in segments:
                bin = segment['length_bin']
                token_count = segment['num_tokens']
                bin_tokens[bin].append(token_count)
    from scipy.stats import describe

    for bin, lengths in bin_tokens.items():
        print("Bin: ", bin)
        print(describe(lengths))

def analyze_tc_data(args):

    splits = ['train' ] # , 'valid_freq', 'test_freq']

    for split in splits:
        split_file = f'{split}_{args.anno_file_suffix}'
        split_file_path = os.path.join(args.data_path, split_file)

        with open(split_file_path, 'r') as split_f:
            split_data = json.load(split_f)
        print("Split:", split)
        analyze_split(split_data)

def analyze_split_knowledge_entitylinking(split_reading_set):

    entity_wiki_lead_section_data = {}
    entity_summarized_wiki_data = {}

    entity_fun_facts_data = defaultdict(set)
    article_data = defaultdict(set)
    fun_fact_entity_map = {}
    article_sentence_entity_map = {}

    for conv_id, conv_data in split_reading_set.items():

        agent1 = conv_data["agent_1"]
        agent2 = conv_data["agent_2"]


        for agent in [agent1, agent2]:

            for fs in ["FS1", "FS2", "FS3"]:
                fact_set = agent[fs]

                entity = fact_set["entity"]

                if "shortened_wiki_lead_section" in fact_set :
                    wiki = fact_set["shortened_wiki_lead_section"]

                    entity_metadata = extract_entity_info(wiki) if "dbpedia_entities" in wiki else {}

                    entity_wiki_lead_section_data[entity] = {
                        "entity": entity,
                        "text": wiki["text"]
                    }

                    entity_wiki_lead_section_data[entity].update(entity_metadata)

                if "fun_facts" in fact_set:

                    facts = fact_set["fun_facts"]

                    for fact in facts:
                        if fact["text"] not in entity_fun_facts_data[entity] :
                            entity_fun_facts_data[entity].add(fact["text"])
                            entity_metadata = extract_entity_info(fact) if "dbpedia_entities" in fact else {}

                            fun_fact_entity_map[fact["text"]] = {
                                "entity": entity,
                                "text": fact["text"],
                            }

                            fun_fact_entity_map[fact["text"]].update(entity_metadata)

                if "summarized_wiki_lead_section" in fact_set:
                    wiki = fact_set["summarized_wiki_lead_section"]

                    entity_metadata = extract_entity_info(wiki) if "dbpedia_entities" in wiki else {}

                    entity_summarized_wiki_data[entity] = {
                        "entity": entity,
                        "text": wiki["text"]
                    }

                    entity_summarized_wiki_data[entity].update(entity_metadata)

        article = conv_data["article"]

        if "AS1" in article:
            for as_i in ["AS1", "AS2", "AS3", "AS4"]:
                sentence = article[as_i]

                if sentence["text"] not in article_data[article["url"]] :
                    article_data[article["url"]].add(sentence["text"])
                    entity_metadata = extract_entity_info(sentence) if "dbpedia_entities" in sentence else {}
                    article_sentence_entity_map[sentence["text"]] = {
                        "url": article["url"],
                        "text": sentence["text"]
                    }
                    article_sentence_entity_map[sentence["text"]].update(entity_metadata)


    wiki_lead_dataframe = pd.DataFrame(list(entity_wiki_lead_section_data.values()), columns=[
            "entity", "text", "num_entities", "normalized_entities", "surface_forms",
            "entity_types", "contains_person", "contains_org", "contains_place",
            "contains_work", "contains_animal"])

    summarized_wiki_dataframe = pd.DataFrame(list(entity_summarized_wiki_data.values()),
                                             columns=[
                                                 "entity", "text", "num_entities", "normalized_entities",
                                                 "surface_forms", "entity_types",
                                                 "contains_person", "contains_org", "contains_place",
                                                 "contains_work", "contains_animal"
                                             ])
    fun_fact_dataframe = pd.DataFrame(list(fun_fact_entity_map.values()),
                                      columns=[
                                          "entity", "text", "num_entities", "normalized_entities",
                                          "surface_forms", "entity_types",
                                          "contains_person", "contains_org", "contains_place",
                                          "contains_work", "contains_animal"
                                      ])
    article_dataframe = pd.DataFrame(list(article_sentence_entity_map.values()),
                                     columns=[
                                         "url", "text", "num_entities", "normalized_entities",
                                         "surface_forms", "entity_types",
                                         "contains_person", "contains_org", "contains_place",
                                         "contains_work", "contains_animal"
                                     ]
                                     )
    return wiki_lead_dataframe, summarized_wiki_dataframe, fun_fact_dataframe, article_dataframe


def extract_entity_info(item):
    entities = item["dbpedia_entities"]
    normed_entities = [e["normalized_entity"] for e in entities]
    surface_forms = [e["surface_form"] for e in entities]
    entity_types = [e["entity_type_dbpedia"][0] if e["entity_type_dbpedia"] else "null" for e in entities]
    person_detected = any("Person" in e["entity_type_dbpedia"] for e in entities if e["entity_type_dbpedia"])
    org_detected = any("Organization" in e["entity_type_dbpedia"] for e in entities if e["entity_type_dbpedia"])
    place_detected = any("Place" in e["entity_type_dbpedia"] for e in entities if e["entity_type_dbpedia"])
    work_detected = any("Work" in e["entity_type_dbpedia"] for e in entities if e["entity_type_dbpedia"])
    animal_detected = any("Animal" in e["entity_type_dbpedia"] for e in entities if e["entity_type_dbpedia"])


    return {
        "num_entities": len(entities),
        "normalized_entities": normed_entities,
        "surface_forms": surface_forms,
        "entity_types": entity_types,
        "contains_person": person_detected,
        "contains_org": org_detected,
        "contains_place": place_detected,
        "contains_work": work_detected,
        "contains_animal": animal_detected
    }


def analyze_entitylinking_in_knowledge(args):

    splits = [
        'train',
        'valid_freq',
        'valid_rare',
        'test_freq',
        'test_rare'
    ]

    for split in splits:
        data_path = os.path.join(
            args.data_path,
            f'{split}_spotlight.json'
        )
        with open(data_path, 'r') as split_knowledge_file:
            split_reading_set = json.load(split_knowledge_file)
        wiki_lead_dataframe, summarized_wiki_dataframe, fun_fact_dataframe, article_dataframe = analyze_split_knowledge_entitylinking(split_reading_set)

        wiki_lead_dataframe.to_csv(f'analysis_results/{split}_wiki_lead_entities.csv')
        summarized_wiki_dataframe.to_csv(f'analysis_results/{split}_summarized_wiki_entities.csv')
        fun_fact_dataframe.to_csv(f'analysis_results/{split}_fun_fact_entities.csv')
        article_dataframe.to_csv(f'analysis_results/{split}_article_entities.csv')
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./tc_processed')
    parser.add_argument('--anno_file_suffix', type=str,
                        default='anno_length_bin.json')
    args = parser.parse_args()
    analyze_entitylinking_in_knowledge(args)

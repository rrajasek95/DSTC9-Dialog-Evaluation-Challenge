import requests, re, json

"""
Author: Wen Cui

To host your own service, there are different ways here https://github.com/dbpedia-spotlight/dbpedia-spotlight/wiki/Installation
Recommend to install from Docker
"""

class SpotlightTagger(object):
    def __init__(self,
                 ontology_json='ontology_classes.json',
                 spotlight_server_url='http://54.166.110.16:8080/rest/annotate' # Athena Production server
    ):

        with open(ontology_json, 'r') as json_file:
            self.ontology = json.load(json_file)

        self.url = spotlight_server_url

        # sport can recognize but with none type
        self.sports_none_type = {'Archery', 'Cycling', 'Fencing', 'Gaelic games', 'Florida Gators football',
                                 'Olympic weightlifting', 'Swimming (sport)', 'Scuba diving'}

        self.video_games_none_type = {'The Legend of Zelda', 'Star Trek', 'Lego Batman', 'Mario Party',
                                      'Motor vehicle theft', 'Animal Crossing', 'Super Smash Bros.', 'Angry Birds'}

        self.tvseries_none_type = {'Outer Banks', 'Star Trek'}

        self.movies_none_type = {'Lego Ninjago', 'Bumblebee', 'Cinderella'}

        self.singer_none_type = {'Cardi B', 'Billie Eilish', 'TheFatRat'}

        self.false_possitive_list = {'Wide area network', 'Alexa Internet', 'Haha (entertainer)', 'Go (game)', 'Ha-ha',
                          'Hectare', 'Hell', 'Good Question', 'Uh-huh', 'Penis', 'Oh Yeah (Yello song)', 'Good Movie', 'Pay Attention',
                        'Fuck', 'Flatulence', 'Blah Blah Blah (Kesha song)', 'Okey', 'Watching Movies', 'Good Stuff',
                          'Vijay Award for Favourite Director', "I Love Music (The O'Jays song)", 'List of recurring Futurama characters',
                          'Hear Music', 'Nice', 'GOOD Music', 'Hell Yeah (Rev Theory song)', 'Anyone', 'Gay'}

    def get_spotlight_annotation(self, text, confidence=0.5):

        headers = {'Accept': 'application/json'}
        data = {'confidence': confidence,
                'text': text}

        response = ""

        try:
            r = requests.post(self.url, headers=headers, data=data, timeout=1)
            if r.status_code == 200:
                response = r.json()
            else:
                print("Spotlight ec2 module bad request of DATA: {} CODE:{}".format(text, r.status_code))
                # self.logger.info("Spotlight ec2 module bad request of DATA: {} CODE:{}".format(text, r.status_code))
                try:
                    r = requests.post('http://api.dbpedia-spotlight.org/en/annotate', headers=headers, data=data, timeout=1)
                    if r.status_code == 200:
                        response = r.json()
                    else:
                        print("Spotlight web service bad request of DATA: {} CODE:{}".format(text, r.status_code))
                        # self.logger.info(
                        # "Spotlight web service bad request of DATA: {} CODE:{}".format(text, r.status_code))
                        return None
                except Exception as e:
                    print("Exception: Spotlight web service module had exception:{} DATA: {}".format(e, text))
                    # self.logger.info("Exception: Spotlight web service module had exception:{} DATA: {}".format(e, text))
        except Exception as e:
            print("Exception: Spotlight ec2 module had exception:{} DATA: {}".format(e, text))
            # self.logger.info(
            #     "Exception: Spotlight ec2 module had exception:{} DATA: {}".format(e, text))
            return None

        if 'Resources' not in response:
            return None
        result = []
        found_entity = set()
        for e in response['Resources']:
            entity = e['@URI'].replace('http://dbpedia.org/resource/', "")
            entity = " ".join(entity.split('_'))
            entity_type_list = e['@types'].split(',')

            entity_type_wikidata = []
            entity_type_schema = []
            entity_type_dbpedia = []

            # Remove duplicates and FP
            if entity in found_entity or entity in self.false_possitive_list:
                print('Filter entity:{}'.format(entity))
                continue

            # Get types from different ontology
            for e_type in entity_type_list:
                if e_type.startswith('Wikidata'):
                    entity_type_wikidata.append(e_type.replace('Wikidata:', ''))
                elif e_type.startswith('Schema'):
                    entity_type_schema.append(e_type.replace('Schema:', ''))
                elif e_type.startswith('DBpedia'):
                    entity_type_dbpedia.append(e_type.replace('DBpedia:', ''))
            entity_type_dbpedia = self.get_sorted_dbpedia_ontology(entity_type_dbpedia)

            surface_form = e['@surfaceForm']
            span = self.get_span(text, surface_form)

            info = {'surface_form': e['@surfaceForm'],
                    'normalized_entity': entity,
                    'entity_type_wikidata': entity_type_wikidata if len(entity_type_wikidata) > 0 else None,
                    'entity_type_schema': entity_type_schema if len(entity_type_schema) > 0 else None,
                    'entity_type_dbpedia': entity_type_dbpedia if len(entity_type_dbpedia) > 0 else None,
                    'similarityScore': e['@similarityScore'],
                    # 'gender': None,
                    'source': 'spotlight',
                    'span': span,
                    # 'akg_id': None
                    }

            found_entity.add(entity)
            result.append(info)
        corrected_result = self.correct_spotlight_el(result)
        return corrected_result if corrected_result else None

    def get_sorted_dbpedia_ontology(self, type_list):
        """
        Sort the type according to dbpedia ontology http://mappings.dbpedia.org/server/ontology/classes/
        from leaf to root
        :param type_list: unsorted type list returned by spotlight
        :return: sorted list [leaf -> root]
        """
        if len(type_list) == 0:
            return type_list
        result = []
        for t in type_list:
            if t in ['Location', 'Wikidata:Q11424']:
                # These two are for sure spotlight will return and not in dbpedia ontology
                continue
            if t in self.ontology:
                result.append((t, self.ontology[t][0]))
            # in case any change of ontology or unexpected value
            else:
                result.append((t, -1))
        # sort by the depth s.t the leaf is in front of the list
        result = sorted(result, key=lambda x: x[1], reverse=True)
        return [r[0] for r in result]

    def get_span(self, text, surface_form):
        """
        Find the first occurrence of span of the entity surface form
        :param text: str
        :param surface_form: str
        :return: a list of [{'start': idx, 'end': idx}] or None
        """
        retval = []
        if surface_form:
            retval = [{'start': m.start(), 'end': m.end()} for m in re.finditer(r'\b{}\b'.format(surface_form), text)]
        return retval if retval else None


    def correct_spotlight_el(self, ner_result):
        retval = ner_result

        # correct none type
        if ner_result and isinstance(ner_result, list):
            for ner in ner_result:
                # Can recognize ner but with None type
                if ner.get('normalized_entity') in self.sports_none_type:
                    ner['entity_type_dbpedia'] = ['Sport', 'Activity']
                    ner['source'] = 'spotlight_correction'
                    ner['entity_type_wikidata'] = ['Q349', 'Q1914636']
                    ner['entity_type_schema'] = None
                elif ner.get('normalized_entity') in self.video_games_none_type:
                    ner['entity_type_dbpedia'] = ['VideoGame', 'Software', 'Work']
                    ner['source'] = 'spotlight_correction'
                    ner['entity_type_wikidata'] = ['Q7889', 'Q7397', 'Q386724']
                    ner['entity_type_schema'] = ['CreativeWork']
                elif ner.get('normalized_entity') in self.tvseries_none_type:
                    ner['entity_type_dbpedia'] = ['TelevisionShow', 'Work']
                    ner['source'] = 'spotlight_correction'
                    ner['entity_type_wikidata'] = ['Q386724', 'Q15416']
                    ner['entity_type_schema'] = ['CreativeWork']
                elif ner.get('normalized_entity') in self.movies_none_type:
                    ner['entity_type_dbpedia'] = ['Film', 'Work']
                    ner['source'] = 'spotlight_correction'
                    ner['entity_type_wikidata'] = ['Q386724']
                    ner['entity_type_schema'] = ['Movie', 'CreativeWork']
                elif ner.get('normalized_entity') in self.singer_none_type:
                    ner['entity_type_dbpedia'] = ['MusicalArtist', 'Artist', 'Person', 'Agent']
                    ner['source'] = 'spotlight_correction'
                    ner['entity_type_wikidata'] = ['Q43229', 'Q24229398', 'Q215380']
                    ner['entity_type_schema'] = ['Organization', 'MusicGroup']
        return retval
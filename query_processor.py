class QueryProcessor:
    def __init__(self, intent_classifier, ranker, chitchat_responder):
        self.intent_classifier = intent_classifier
        self.ranker = ranker
        self.chitchat_responder = chitchat_responder

    def generate_response(self, query):
        intent = self.intent_classifier.predict_intent(query)
        if intent == 'CHITCHAT':
            response = str(self.chitchat_responder.get_response(query))
            return response
        elif intent == 'IR':
            result_ids = self.ranker.search(query, 3)#, 'lsh_sqlite')
            url_pattern = 'https://stackoverflow.com/questions/{}/\n'
            response = 'Try looking at these links:\n'
            for result_id in result_ids:
                response += url_pattern.format(result_id)
            return response
        else:
            raise RuntimeError('Unkonwn query intent.')
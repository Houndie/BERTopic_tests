from bertopic import BERTopic
import os
import xml.etree.ElementTree

if __name__ == "__main__":
    print(f"loading posts")
    docs = []
    data = xml.etree.ElementTree.parse("data/Posts.xml").getroot()
    for d in data:
        docs.append(d.attrib["Body"])


    print(f"training model")
    seed_topic_list = [["issue", "ticket", "bug", "epic", "jira", "agile", "velocity"], ["security", "alert", "traceback", "log", "error", "warning"]]
    topic_model = BERTopic(seed_topic_list=seed_topic_list).fit(docs)
    topic_model.save("mymodel.model")
    print(f"topics")
    for topic in topic_model.get_topics():
        topic_label = topic_model.topic_labels_.get(topic, "unknown")
        print(f"{topic_label}")

    print(f"transforming test string")
    topics, probs = topic_model.transform("list all tickets currently assigned to me")
    for topic in topics:
        topic_label = topic_model.topic_labels_.get(topic, "unknown")
        print(f"{topic_label}")

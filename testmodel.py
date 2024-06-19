"""Test the model on a specific string"""

from bertopic import BERTopic
import os
import xml.etree.ElementTree
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "content",
        nargs="?",
        type=str,
        help="The message to categorize (default: %(default)s)",
    )
    args = parser.parse_args()

    print("loading model")
    topic_model = BERTopic().load("mymodel.model")

    print(f"transforming test string")
    topics, probs = topic_model.transform(args.content)
    for topic in topics:
        topic_label = topic_model.topic_labels_.get(topic, "unknown")
        print(f"{topic_label}")

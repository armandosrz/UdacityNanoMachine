def count_words(s, n):
    """Return the n most frequently occuring words in s."""

    # TODO: Count the number of occurences of each word in s
    words = s.split(' ')
    occurences = []
    for word in words:
        occurences.append((word, words.count(word)))
        words = filter(lambda a: a != word, words)


    # TODO: Sort the occurences in descending order (alphabetically in case of ties)
    occurences = sorted(occurences, reverse=True, key=get_key)
    # TODO: Return the top n words as a list of tuples (<word>, <count>)
    return occurences[:n]


def test_run():
    """Test count_words() with some inputs."""
    print count_words("cat bat mat cat bat cat", 3)
    print count_words("betty bought a bit of butter but the butter was bitter", 3)

def get_key(item):
    return item[1], [-ord(c) for c in item[0]]

if __name__ == '__main__':
    test_run()

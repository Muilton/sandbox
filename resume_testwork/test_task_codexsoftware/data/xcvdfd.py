def flatten(dictionary, prefix='', result={}):
    for key in dictionary.keys():
        if isinstance(dictionary[key], dict):  # если значение является словарем
            if dictionary[key]:
                if prefix:
                    newkey = f"{prefix}/{key}"
                else:
                    newkey = key

                flatten(dictionary[key], newkey)
            else:  # пустой словарь
                if prefix:
                    result.update({f"{prefix}/{key}": ""})
                else:
                    result.update({key: ""})
        else:  # не словарь
            if prefix:
                result.update({f"{prefix}/{key}": dictionary[key]})
            else:
                result.update({key: dictionary[key]})
    print(result)
    return result


if __name__ == '__main__':
    # test_input = {"key": {"deeper": {"more": {"enough": "value"}}}}
    # print(' Input: {}'.format(test_input))
    # print('Output: {}'.format(flatten(test_input)))

    # These "asserts" using only for self-checking and not necessary for auto-testing
    assert flatten({"key": "value"}) == {"key": "value"}
    assert flatten(
        {"key": {"deeper": {"more": {"enough": "value"}}}}
    ) == {"key/deeper/more/enough": "value"}
    assert flatten({"empty": {}}) == {"empty": ""}, "Empty value"
    assert flatten({"name": {
        "first": "One",
        "last": "Drone"},
        "job": "scout",
        "recent": {},
        "additional": {
            "place": {
                "zone": "1",
                "cell": "2"}}}
    ) == {"name/first": "One",
          "name/last": "Drone",
          "job": "scout",
          "recent": "",
          "additional/place/zone": "1",
          "additional/place/cell": "2"}
    print('You all set. Click "Check" now!')
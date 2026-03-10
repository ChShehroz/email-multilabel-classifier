def build_targets(y2, y3, y4):

    t2 = y2

    t23 = y2 + "||" + y3

    t234 = y2 + "||" + y3 + "||" + y4

    return {
        "t2": t2,
        "t23": t23,
        "t234": t234
    }
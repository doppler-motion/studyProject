"""
题目描述
定义：开头和结尾都是元音字母（aeiouAEIOU）的字符串为 元音字符串 ，其中混杂的非元音字母数量为其 瑕疵度 。比如:
·	“a” 、 “aa”是元音字符串，其瑕疵度都为0
·	“aiur”不是元音字符串（结尾不是元音字符）
·	“abira”是元音字符串，其瑕疵度为2
给定一个字符串，请找出指定瑕疵度的最长元音字符子串，并输出其长度，如果找不到满足条件的元音字符子串，输出0。
子串：字符串中任意个连续的字符组成的子序列称为该字符串的子串。
解答要求
时间限制：1000ms, 内存限制：256MB
输入
首行输入是一个整数，表示预期的瑕疵度flaw，取值范围 [0, 65535]。接下来一行是一个仅由字符a-z和A-Z组成的字符串，字符串长度 (0, 65535]。
输出
输出为一个整数，代表满足条件的元音字符子串的长度。
样例
输入样例 1
0asdbuiodevauufgh
输出样例 1
3
提示样例 1
满足条件的最长元音字符子串有两个，分别为uio和auu，长度为3。
输入样例 2
2aeueo
输出样例 2
0
提示样例 2
没有满足条件的元音字符子串，输出0
输入样例 3
1aabeebuu

"""


def findcSubStr(string, flaw):
    base_str = "aeiouAEIOU"
    ans = 0
    begin, n = 0, len(string)
    flaw_num = 0
    while begin < n:
        if string[begin] in base_str:
            for i in range(begin, n):
                if string[i] in base_str:
                    ans = max(i - begin + 1, ans)
                else:
                    if flaw != 0:
                        flaw_num += 1

                        if flaw_num == flaw:
                            break
                    else:
                        break
        else:
            flaw_num = 0

        begin += 1

    if flaw_num != flaw:
        return 0

    return ans


def findSubStr1(string, flaw):
    base_str = "aeiouAEIOU"
    ans = 0
    for i in range(len(string)):
        if string[i] in base_str:
            cur_len = 0
            err_len = 0
            for j in range(i, len(string)):
                if string[j] in base_str:
                    cur_len += 1
                else:
                    err_len += 1

                if err_len > flaw:
                    break
                ans = max(ans, cur_len)

    return ans if ans > 0 else 0



if __name__ == "__main__":
    test_str = "aabeebuu"
    flaws = 1
    print(findcSubStr(test_str, flaws))

class Solution:
    def oddString(self, words) -> str:
        ans = [[]] * len(words)
        print(ans)
        for i, item in enumerate(words):
            for j in range(1, len(item)):
                ans[i].append(ord(item[j]) - ord(item[j - 1]))

        print(ans)
        if ans[0] == ans[1]:
            for i in range(2, len(ans)):
                if ans[0] != ans[i]:
                    return words[i]
        return words[1] if ans[0] == ans[2] else words[0]


if __name__ == "__main__":
    string_list = ["adc", "wzy", "abc"]
    s = Solution()
    print(s.oddString(string_list))

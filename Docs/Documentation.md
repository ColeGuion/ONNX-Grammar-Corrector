# GEC Program Documentation and Info

### Skipping repeated tokens
- As we are fetch our next most likely tokens with **`GetMaxTokens()`** we call **`checkRepeating()`** on each sequence of tokens
- **`checkRepeating()`** checks if the latest 6 values in the given array are all the same or alternating (e.g. [5, 8, 5, 8, 5, 8])
- If so we denote those repeating tokens and ignore them when they come up as the next most likely token
  - So we select the next most likely token that has not been previously repeating for that sequence
  - For example, we would select any token that is not a 5 or an 8
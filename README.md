# LongSumm


| Folder Name             | Description |
|-------------------------|-------------|
| `abstractive_summaries` | Contains JSON files with blog summaries of the scientific papers. The summaries are divided by the word count that is clustered into bins of size 100.  |
| `papers`                | Contains the actual scientific papers in both JSON and PDF formats. The JSON files may include structured information that is parsed by the paper. |

## Usage
- **Key**: File names of each of the papers inside `papers` directory is the record id. This is matched with the `id` field in the summary json inside `abstractive_summaries` directory.
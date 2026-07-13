class Tokenizer
{
public:
    bool load(const std::string& tokenizer_json_path);

    std::vector<int> encode(const std::string& text);

    std::string decode(const std::vector<int>& ids);

private:
    std::unique_ptr<Normalizer> normalizer_;
    std::unique_ptr<PreTokenizer> pretokenizer_;
    std::unique_ptr<BPETokenizer> bpe_;
    std::unique_ptr<PostProcessor> postprocessor_;
    std::unique_ptr<Decoder> decoder_;
};
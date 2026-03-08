import sglang as sgl

def main():
    llm = sgl.Engine(model_path="/home/wxt/LLaDA2.0-mini",
                     dllm_algorithm="LowConfidenceFDFO",
                     attention_backend="flashinfer",
                     max_running_requests=1, disable_cuda_graph=True,
                     trust_remote_code=True)

    prompts = [
        "<role>SYSTEM</role>detailed thinking off<|role_end|><role>HUMAN</role>简要介绍长城<|role_end|><role>ASSISTANT</role>"
    ]

    sampling_params = {
        "temperature": 0,
        "max_new_tokens": 1024,
    }

    outputs = llm.generate(prompts, sampling_params)
    print(outputs)

if __name__ == '__main__':
    main()
import torch
import os
from datautils import get_loaders
from tqdm import tqdm
import torch.nn as nn

@torch.no_grad()
def evaluate(model, args):
    results = {}

    raw_length = model.seqlen
    model.seqlen = 2048

    # import code
    # code.interact(local=locals())

    # if "opt" in args.net.lower():
    #     model.model.decoder = model.model.decoder.to(model.device)
    # elif "llama" in args.net.lower() or "mixtral" in args.net.lower():
    #     model = model.to(model.device)
    # elif "falcon" in args.net.lower():
    #     model.transformer = model.transformer.to(model.device)

    # if args.eval_ppl:
    # for dataset in ["wikitext2", "ptb", "c4","ptb-new",'c4-new']:
    # for dataset in ["c4", "ptb", "wikitext2"]:
    for dataset in ["wikitext2"]:
        cache_testloader = f'{args.cache_dir}/testloader_{args.model_family}_{dataset}_all.cache'
        if os.path.exists(cache_testloader):
            testloader = torch.load(cache_testloader)
        else:
            dataloader, testloader = get_loaders(
                dataset,
                seed=args.seed,
                model=args.model,
                seqlen=model.seqlen,
            )
            torch.save(testloader, cache_testloader)
        if "c4" in dataset:
            testenc = testloader
        else:
            testenc = testloader.input_ids

        nsamples = testenc.numel() // model.seqlen
        use_cache = model.config.use_cache
        model.config.use_cache = False
        model.eval()
        nlls = []
        # torch.random.manual_seed(123)
        # num_test = min(nsamples, 512)
        model.lm_head = model.lm_head.to('cuda')
        total_nll = 0.0
        total_tokens = 0
        with torch.inference_mode():
            for i in tqdm(range(nsamples)):
                # i = torch.randint(0, nsamples, (1,))
                # for i in tqdm(range(nsamples)):
                batch = testenc[:, (i * model.seqlen): ((i + 1) * model.seqlen)].to(model.device)
                with torch.no_grad():
                    if "opt" in args.net.lower():
                        outputs = model.model.decoder(batch)
                    elif "llama" in args.net.lower() or "mixtral" in args.net.lower():
                        outputs = model.model(batch)
                    elif "falcon" in args.model:
                        outputs = model.transformer(batch)
                hidden_states = outputs[0]

                logits = model.lm_head(hidden_states)
                shift_logits = logits[:, :-1, :]
                shift_labels = testenc[:, (i * model.seqlen): ((i + 1) * model.seqlen)][
                               :, 1:
                               ].to(model.lm_head.weight.device)

                loss_fct = nn.CrossEntropyLoss(reduction='sum')
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                total_nll += loss.item()
                total_tokens += shift_labels.numel()

                # loss_fct = nn.CrossEntropyLoss()
                # loss = loss_fct(
                #     shift_logits.reshape(-1, shift_logits.size(-1)),
                #     shift_labels.reshape(-1),
                # )
                # neg_log_likelihood = loss.float() * model.seqlen
                # nlls.append(neg_log_likelihood)

                print(loss.item())


        torch.cuda.empty_cache()
        # ppl = torch.exp(torch.stack(nlls).sum() / (len(nlls) * model.seqlen))
        ppl = torch.exp(torch.tensor(total_nll / total_tokens))
        print(f'{dataset} : {ppl.item()}')
        model.config.use_cache = use_cache
        results[dataset] = ppl.item()

    model.seqlen = raw_length

    return results
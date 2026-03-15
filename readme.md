For different horizons:
train_loader = DataLoader(TSDataset(train_slice, seq_len=seq_len, pred_len=pred_len), batch_size=batch_size, shuffle=True, drop_last=True,
                          generator=g, worker_init_fn=lambda w: np.random.seed(SEED + w))

val_loader   = DataLoader(TSDataset(val_slice, seq_len=seq_len, pred_len=pred_len), batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(TSDataset(test_slice, seq_len=seq_len, pred_len=pred_len), batch_size=batch_size, shuffle=False)

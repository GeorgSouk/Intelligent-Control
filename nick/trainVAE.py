import torch
from torch.utils.data import DataLoader
import torch.nn as nn

# Assume these are defined elsewhere:
# - dataset: PyTorch Dataset yielding (s_t, s_{t+1}, a)
# - encoder: maps (s_t, s_{t+1}) -> (mu, logvar)
# - decoder: maps (s_t, z) -> \hat{s}_{t+1}
# - prior_net: maps s_t -> (mu0, logvar0)
# - expert_policy: supports .log_prob(a, given_state=s_t)

class Autoencoder(nn.Module):
	def __init__(self, input_dim=6, latent_dim=1):
		super().__init__()
		# Encoder: s_t + s_t+1 →  latent representation z (i think its the torque) 
		self.encoder = nn.Sequential(
			nn.Linear(input_dim, 128),
			nn.ReLU(),
			nn.Linear(128, 128),
			nn.ReLU(),
			nn.Linear(128, latent_dim),
		)
		# Decoder: z → approximation s_t+1 
		self.decoder = nn.Sequential(
			nn.Linear(latent_dim, 128),
			nn.ReLU(),
   			nn.Linear(128, 128),
			nn.ReLU(),
			nn.Linear(128, 3),
			nn.Sigmoid(),              
		)

	def forward(self, x):
		
		z = self.encoder(x)
		x_hat = self.decoder(z)
		return x_hat, z

# Hyperparameters
batch_size = 64
num_epochs = 100
learning_rate = 1e-3
beta = 1.0  # weight on expert KL term
lambda_recon = 1.0  # precision for reconstruction term

# DataLoader
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

encoder = Autoencoder(input_dim=6, latent_dim=1).encoder
decoder = Autoencoder(input_dim=6, latent_dim=1).decoder

# Optimizer: update encoder, decoder, prior_net (and optionally expert_policy if learning it)
params = list(encoder.parameters()) + list(decoder.parameters()) + list(prior_net.parameters())
optimizer = torch.optim.Adam(params, lr=learning_rate)

# Loss function defined as before
def vae_loss(s, s_next, a, encoder, decoder, prior_net, expert_policy, beta, lambda_recon):
    # Encode posterior q_phi(z|s, s_next)
    mu, logvar = encoder(s, s_next)
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    z = mu + eps * std  # reparameterization trick

    # Reconstruction
    s_next_hat = decoder(s, z)
    recon_loss = ((s_next_hat - s_next).pow(2).view(s.size(0), -1).sum(dim=1).mean())
    recon_term = lambda_recon * recon_loss

    # Prior KL: KL(q(z|s,s') || p(z|s))
    mu0, logvar0 = prior_net(s)
    kl_prior = 0.5 * (
        logvar0 - logvar - 1
        + (torch.exp(logvar) + (mu - mu0).pow(2)) / torch.exp(logvar0)
    ).sum(dim=1).mean()

    # Expert KL: KL(q(z|s,s') || pi_E(a|s))
    # log q(z|.) up to constant
    log_q = -0.5 * (logvar + (z - mu).pow(2) / torch.exp(logvar)).sum(dim=1)
    # log pi_E(a | s)
    log_pi = expert_policy.log_prob(a, given_state=s)
    kl_expert = (log_q - log_pi).mean()

    # Total loss (we minimize negative ELBO)
    loss = recon_term + kl_prior + beta * kl_expert
    return loss, recon_loss.detach(), kl_prior.detach(), kl_expert.detach()

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder.to(device)
decoder.to(device)
prior_net.to(device)
# expert_policy may also need to be on device

for epoch in range(1, num_epochs + 1):
    encoder.train()
    decoder.train()
    prior_net.train()
    total_loss = 0.0
    total_recon = 0.0
    total_kl_prior = 0.0
    total_kl_expert = 0.0

    for batch in dataloader:
        s, s_next, a = batch
        s, s_next, a = s.to(device), s_next.to(device), a.to(device)

        optimizer.zero_grad()
        loss, recon, kl_p, kl_e = vae_loss(
            s, s_next, a,
            encoder, decoder, prior_net,
            expert_policy, beta, lambda_recon
        )
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon.item()
        total_kl_prior += kl_p.item()
        total_kl_expert += kl_e.item()

    avg_loss = total_loss / len(dataloader)
    avg_recon = total_recon / len(dataloader)
    avg_kl_prior = total_kl_prior / len(dataloader)
    avg_kl_expert = total_kl_expert / len(dataloader)

    print(f"Epoch {epoch}/{num_epochs} | Loss: {avg_loss:.4f}"
          f" | Recon: {avg_recon:.4f} | KL_prior: {avg_kl_prior:.4f}"
          f" | KL_expert: {avg_kl_expert:.4f}")

# After training, switch to evaluation mode:
encoder.eval()
decoder.eval()
prior_net.eval()

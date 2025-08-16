import torch
import torch.nn as nn
from torch import autograd
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR


class Encoder(nn.Module):
    def __init__(self, latent_dim=256, filter_size=64, device='cuda'):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.filter_size = filter_size
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Encoder: 256x256 -> 4x4 feature map (filter_size * 8 channels)
        self.encoder = nn.Sequential(
            # 256x256 -> 128x128
            nn.Conv2d(1, filter_size, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 128x128 -> 64x64
            nn.Conv2d(filter_size, filter_size, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(filter_size),
            nn.LeakyReLU(0.2, inplace=True),

            # 64x64 -> 32x32
            nn.Conv2d(filter_size, filter_size * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(filter_size * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # 32x32 -> 16x16
            nn.Conv2d(filter_size * 2, filter_size * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(filter_size * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # 16x16 -> 8x8
            nn.Conv2d(filter_size * 2, filter_size * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(filter_size * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # 8x8 -> 4x4
            nn.Conv2d(filter_size * 4, filter_size * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(filter_size * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Project feature map to latent space
        self.fc = nn.Linear(filter_size * 8 * 4 * 4, latent_dim)

        self.to(self.device)

    def forward(self, x):
        x = self.encoder(x)
        # Flatten the feature map
        x = x.view(x.size(0), -1)
        # Project to latent space
        z = self.fc(x)
        return z


class Generator(nn.Module):
    def __init__(self, latent_dim=256, filter_size=64, device='cuda'):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.filter_size = filter_size
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # FC-Generator: latent_dim -> 4x4 feature map (filter_size * 8 channels)
        self.fc = nn.Linear(latent_dim, filter_size * 8 * 4 * 4)
        
        # Decoder: 4x4 -> 256x256
        self.decoder = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(filter_size * 8, filter_size * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(filter_size * 4),
            nn.ReLU(True),

            # 8x8 -> 16x16
            nn.ConvTranspose2d(filter_size * 4, filter_size * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(filter_size * 2),
            nn.ReLU(True),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(filter_size * 2, filter_size * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(filter_size * 2),
            nn.ReLU(True),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(filter_size * 2, filter_size, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(filter_size),
            nn.ReLU(True),
            
            # 64x64 -> 128x128
            nn.ConvTranspose2d(filter_size, filter_size, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(filter_size),
            nn.ReLU(True),

            # 128x128 -> 256x256
            nn.ConvTranspose2d(filter_size, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )
        
        self.to(self.device)

    def forward(self, z):
        # Project noise to feature map
        x = self.fc(z)
        # Reshape to 4x4 feature map
        x = x.view(x.size(0), self.filter_size * 8, 4, 4)
        # Generate image
        return self.decoder(x)


class Discriminator(nn.Module):
    def __init__(self, latent_dim=256, filter_size=64, device='cuda'):
        super(Discriminator, self).__init__()
        self.filter_size = filter_size
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Encoder: 256x256 -> 4x4 (Use instance norm instead of batch norm, which is not recommended for WGAN-GP)
        self.encoder = nn.Sequential(
            # 256x256 -> 128x128
            nn.Conv2d(1, filter_size, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 128x128 -> 64x64
            nn.Conv2d(filter_size, filter_size, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(filter_size),
            nn.LeakyReLU(0.2, inplace=True),

            # 64x64 -> 32x32
            nn.Conv2d(filter_size, filter_size * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(filter_size * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # 32x32 -> 16x16
            nn.Conv2d(filter_size * 2, filter_size * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(filter_size * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # 16x16 -> 8x8
            nn.Conv2d(filter_size * 2, filter_size * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(filter_size * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # 8x8 -> 4x4
            nn.Conv2d(filter_size * 4, filter_size * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(filter_size * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # FC-Generator: Upsample z
        self.fc_z = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # FC-Intermediate: Concatenate features and latent vector
        self.fc_inter = nn.Sequential(
            nn.Linear(filter_size * 8 * 4 * 4 + 512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3)
        )

        # FC-Output: Intermediate to logit
        self.fc_out = nn.Linear(1024, 1)

        self.to(self.device)

    def forward(self, x, z):
        features = self.encoder(x)
        features = features.view(features.size(0), -1)
        latent = self.fc_z(z)

        # Concatenate features and latent vector as tuple
        tup = torch.cat([features, latent], dim=1)

        intermediate = self.fc_inter(tup)
        output = self.fc_out(intermediate)
        return output, intermediate


class WGAN_EGBAD:
    def __init__(self, g_lr=2e-4, e_lr=2e-4, d_lr=2e-4, betas=(0.5, 0.999), filter_size=64, latent_dim=256, lambda_gp=10,
                 anomaly_threshold=0.5, device='cuda'):
        self.anomaly_threshold = anomaly_threshold
        self.latent_dim = latent_dim
        self.lambda_gp = lambda_gp
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Initialize models
        self.G = Generator(latent_dim=latent_dim, filter_size=filter_size, device=device)
        self.D = Discriminator(latent_dim=latent_dim, filter_size=filter_size, device=device)
        self.E = Encoder(latent_dim=latent_dim, filter_size=filter_size, device=device)

        # Initialize weights
        self._init_weights()

        # Betas of (0.5, 0.999) are commonly used in GANs
        self.D_optim = optim.Adam(self.D.parameters(), lr=d_lr, betas=betas)
        self.G_optim = optim.Adam(self.G.parameters(), lr=g_lr, betas=betas)
        self.E_optim = optim.Adam(self.E.parameters(), lr=e_lr, betas=betas)
        
        # Learning rate schedulers
        self.D_scheduler = ExponentialLR(self.D_optim, gamma=0.98)
        self.G_scheduler = ExponentialLR(self.G_optim, gamma=0.98)
        self.E_scheduler = ExponentialLR(self.E_optim, gamma=0.98)
        
        # Discriminator outputs a logit
        self.bce = nn.BCEWithLogitsLoss()

    def _init_weights(self):
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
        
        self.G.apply(init_func)
        self.E.apply(init_func)
        self.D.apply(init_func)

    # Calculate gradient penalty for WGAN-GP (adapted for the BiGAN architecture)
    # The gradient penalty enforces the Lipschitz constraint on the discriminator
    # GP = λ * E_x[(||∇_x D(x)||_2 − 1)^2] (see https://arxiv.org/abs/1704.00028)
    # For the BiGAN we need to consider gradients for both images and latents:
    # GP = λ * E[(||∇_(x,z) D(x,z)||_2 − 1)^2]
    def calc_gradient_penalty(self, disc, real_images, real_z, fake_images, fake_z):
        batch_size = real_images.size(0)
        
        # Sample two random interpolation coefficients (one for images, one for latents)
        alpha_imgs = torch.rand(batch_size, 1, 1, 1, device=self.device)
        alpha_x = alpha_imgs.expand_as(real_images)
        alpha_latent = torch.rand(batch_size, 1, device=self.device)
        alpha_z = alpha_latent.expand_as(real_z)
        
        # Create interpolated samples between real and fake pairs
        interpolated_x = alpha_x * real_images + (1 - alpha_x) * fake_images
        interpolated_z = alpha_z * real_z + (1 - alpha_z) * fake_z
        
        # Interpolated samples require gradients (see GP-Equation)
        interpolated_x = interpolated_x.to(self.device).requires_grad_(True)
        interpolated_z = interpolated_z.to(self.device).requires_grad_(True)
        
        out, _ = disc(interpolated_x, interpolated_z)

        # Compute gradients of discriminator outputs for both inputs (= ∇_(x,z) D(x,z))
        gradients = autograd.grad(
            outputs=out,
            inputs=[interpolated_x, interpolated_z],
            grad_outputs=torch.ones_like(out, device=self.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )
        
        # Flatten gradient tensors
        grad_x = gradients[0].view(batch_size, -1)
        grad_z = gradients[1].view(batch_size, -1)
        
        # Combine gradients and compute L2 norm
        combined_grad = torch.cat([grad_x, grad_z], dim=1)
        grad_norm = combined_grad.norm(2, dim=1)
        
        # Compute gradient penalty => Penalize deviation from norm=1 (Lipschitz constraint)
        gradient_penalty = ((grad_norm - 1) ** 2).mean()
        
        return gradient_penalty

    def train_step(self, real_images, n_critic=5):
        """
        Perform one training step for BiGAN-WGAN-GP
        
        Following WGAN-GP training protocol:
        1. Train critic (discriminator) n_critic times 
        2. Train generator and encoder once
        
        Args:
            real_images: Tensor of shape (batch_size, 1, 256, 256) with values in [-1, 1]
            n_critic: Number of critic updates per generator update (default: 5)
        """
        real_images = real_images.to(self.device)
        batch_size = real_images.size(0)
        
        # Train Discriminator (Critic)
        d_loss_final = 0
        for _ in range(n_critic):
            z_prior = torch.randn(batch_size, self.latent_dim, device=self.device)
            
            self.D_optim.zero_grad()
            
            # Forward passes
            with torch.no_grad():
                z_encoded = self.E(real_images)  # E(x) - encode real images
                x_generated = self.G(z_prior)    # G(z) - generate from prior
            
            DE, _ = self.D(real_images, z_encoded)       # D(x, E(x)) - real pairs
            DG, _ = self.D(x_generated, z_prior)         # D(G(z), z) - fake pairs
            
            critic_real = DE.mean()
            critic_fake = DG.mean()
            
            # Gradient penalty for Lipschitz constraint
            gp = self.calc_gradient_penalty(self.D, real_images, z_encoded, x_generated, z_prior)
            
            # WGAN-GP critic loss: max D(real_pair) - D(fake_pair) = min -(D(real_pair) - D(fake_pair))
            # = min -D(real_pair) + D(fake_pair)
            # Total discriminator loss: min -D(real_pair) + D(fake_pair) + λ * GP
            d_loss = -critic_real + critic_fake + self.lambda_gp * gp
            
            # Backward + Step
            d_loss.backward()
            self.D_optim.step()
            
            d_loss_final = d_loss.item()
        
        # Train Generator and Encoder
        z_prior = torch.randn(batch_size, self.latent_dim, device=self.device)
        
        self.G_optim.zero_grad()
        self.E_optim.zero_grad()
        
        # Forward passes
        z_encoded = self.E(real_images)     # E(x) - encode real images
        x_generated = self.G(z_prior)       # G(z) - generate from prior
        
        DE_ge, _ = self.D(real_images, z_encoded)    # D(x, E(x)) - real pairs
        DG_ge, _ = self.D(x_generated, z_prior)      # D(G(z), z) - fake pairs
        
        # WGAN-GP generator loss: min D(real_pair) - D(fake_pair) = min -D(fake_pair) + D(real_pair)
        critic_real_ge = DE_ge.mean()
        critic_fake_ge = DG_ge.mean()
        adv_loss = -critic_fake_ge + critic_real_ge
        
        # Train also on reconstruction loss
        # Train G+E using feature matching loss (L1 distance between intermediate features from D)
        z_encoded_ge = self.E(real_images)
        x_recon = self.G(z_encoded_ge)
        _, feats_real = self.D(real_images, z_encoded_ge)
        _, feats_recon = self.D(x_recon, z_encoded_ge)
        recon_loss = nn.L1Loss(reduction="sum")(feats_recon, feats_real)
        recon_loss = recon_loss * 5e-5  # Scaled to match adversarial loss

        # Combined G+E loss and scaled reconstruction loss
        ge_loss = adv_loss + recon_loss
        
        # Backward + Step
        ge_loss.backward()
        self.G_optim.step()
        self.E_optim.step()

        return d_loss_final, adv_loss.item(), recon_loss.item()

    def validate_step(self, real_images):
        real_images = real_images.to(self.device)
        batch_size = real_images.size(0)
        
        z_prior = torch.randn(batch_size, self.latent_dim, device=self.device)
        
        with torch.no_grad():
            z_encoded = self.E(real_images)
            x_generated = self.G(z_prior)
        
        DE, _ = self.D(real_images, z_encoded)
        DG, _ = self.D(x_generated, z_prior)
        
        critic_real = DE.mean()
        critic_fake = DG.mean()
        
        gp = self.calc_gradient_penalty(self.D, real_images, z_encoded, x_generated, z_prior)
        d_loss = -critic_real + critic_fake + self.lambda_gp * gp

        adv_loss = -critic_fake + critic_real

        _, feats_real = self.D(real_images, z_encoded)
        _, feats_recon = self.D(self.G(z_encoded), z_encoded)
        recon_loss = nn.L1Loss(reduction="sum")(feats_recon, feats_real)
        recon_loss = recon_loss * 5e-5

        return d_loss.item(), adv_loss.item(), recon_loss.item()

    def generate_samples(self, num_samples=1):
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim, device=self.device)
            fake_images = self.G(z)
            return fake_images

    def save_models(self, checkpoint_path):
        torch.save({
            'discriminator_state_dict': self.D.state_dict(),
            'encoder_state_dict': self.E.state_dict(),
            'generator_state_dict': self.G.state_dict(),
            'd_optimizer_state_dict': self.D_optim.state_dict(),
            'g_optimizer_state_dict': self.G_optim.state_dict(),
            'e_optimizer_state_dict': self.E_optim.state_dict(),
        }, checkpoint_path)

    def load_models(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.G.load_state_dict(checkpoint['generator_state_dict'])
        self.D.load_state_dict(checkpoint['discriminator_state_dict'])
        self.E.load_state_dict(checkpoint['encoder_state_dict'])
        self.D_optim.load_state_dict(checkpoint['d_optimizer_state_dict'])
        self.G_optim.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.E_optim.load_state_dict(checkpoint['e_optimizer_state_dict'])

    def update_schedulers(self):
        """Update learning rate schedulers"""
        self.D_scheduler.step()
        self.G_scheduler.step()
        self.E_scheduler.step()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Test Encoder
    encoder = Encoder(latent_dim=128, filter_size=16, device=device)
    dummy_image = torch.randn(2, 1, 256, 256).to(device)
    latent_vector = encoder(dummy_image)
    print("Encoder Input shape:", dummy_image.shape)
    print("Encoder Output shape:", latent_vector.shape)
    
    # Test Generator
    generator = Generator(latent_dim=128, filter_size=16, device=device)
    dummy_z = torch.randn(2, 128).to(device)
    gen_output = generator(dummy_z)
    print("Generator Input shape:", dummy_z.shape)
    print("Generator Output shape:", gen_output.shape)
    
    # Test Discriminator
    discriminator = Discriminator(latent_dim=128, filter_size=16, device=device)
    dummy_image = torch.randn(2, 1, 256, 256).to(device)
    dummy_latent = torch.randn(2, 128).to(device)
    disc_output = discriminator(dummy_image, dummy_latent)
    print("Discriminator Input shape:", dummy_image.shape)
    print("Discriminator Output shape(s):", disc_output[0].shape, disc_output[1].shape)
    
    # Test WGAN_EGBAD
    wgan_egbad = WGAN_EGBAD(filter_size=32, latent_dim=128, device=device)
    test_batch = torch.randn(4, 1, 256, 256).to(device)
    d_loss, ge_loss, recon_loss = wgan_egbad.train_step(test_batch)
    print(f"Training step - D loss: {d_loss:.4f}, GE loss: {ge_loss:.4f}, Recon loss: {recon_loss:.4f}")
    generated_samples = wgan_egbad.generate_samples(num_samples=4)
    print("Generated samples shape:", generated_samples.shape)

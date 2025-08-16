import torch
import torch.nn as nn
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
        
        # Encoder: 256x256 -> 4x4
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


class EGBAD:
    def __init__(self, g_lr=2e-4, e_lr=2e-4, d_lr=2e-4, betas=(0.5, 0.999), filter_size=64, latent_dim=256,
                 anomaly_threshold=0.5, device='cuda'):
        self.anomaly_threshold = anomaly_threshold
        self.latent_dim = latent_dim
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

    # BiGAN loss: min_G,E max_D V(D, E, G) = E_x[log D(x, E(x))] + E_z[log(1 - D(G(z), z))] (see http://arxiv.org/abs/1605.09782)
    # Discriminator loss
    def D_loss(self, DG, DE):
        # DE should be 1 and DG should be 0 (use label smoothing)
        ones = torch.full_like(DE, 0.9)
        zeros = torch.full_like(DG, 0.1)
        loss = self.bce(DE, ones) + self.bce(DG, zeros)
        return loss

    # Generator/Encoder losses
    def GE_losses(self, DG, DE):
        # DG should be 1 and DE should be 0 (use label smoothing)
        ones = torch.full_like(DG, 0.9)
        zeros = torch.full_like(DE, 0.1)
        g_loss, e_loss = self.bce(DG, ones), self.bce(DE, zeros)
        return g_loss, e_loss

    def train_step(self, real_images):
        """
        Perform one training step for EGBAD following BiGAN architecture
        Args:
            real_images: Tensor of shape (batch_size, 1, 256, 256) with values in [-1, 1]
        """
        real_images = real_images.to(self.device)
        batch_size = real_images.size(0)
        
        # Sample random latent vectors
        z_prior = torch.randn(batch_size, self.latent_dim, device=self.device)

        # Train Discriminator
        self.D_optim.zero_grad()

        # Forward passes to get pairs
        z_encoded = self.E(real_images)  # E(x) - encode real images
        z_encoded_detach = z_encoded.detach()
        x_generated = self.G(z_prior)    # G(z) - generate from prior
        x_generated_detach = x_generated.detach()
        DE, _ = self.D(real_images, z_encoded_detach)  # D(x, E(x))
        DG, _ = self.D(x_generated_detach, z_prior)    # D(G(z), z)

        # Discriminator loss: maximize D(x, E(x)) and minimize D(G(z), z)
        d_loss = self.D_loss(DG, DE)
        
        # Backward + Step
        d_loss.backward()
        self.D_optim.step()
        
        # Train Generator/Encoder jointly
        self.G_optim.zero_grad()
        self.E_optim.zero_grad()

        # Re-compute forward passes
        DE_ge, _ = self.D(real_images, z_encoded)
        DG_ge, _ = self.D(x_generated, z_prior)

        # Generator/Encoder losses: fool the discriminator
        g_loss, e_loss = self.GE_losses(DG_ge, DE_ge)  # Make D(G(z), z) = 1 and D(x, E(x)) = 0

        # Train also on reconstruction loss (inspired by https://www.sciencedirect.com/science/article/pii/S1877050920318445)
        # Train G+E using feature matching loss (L1 distance between intermediate features from D)
        z_encoded_ge = self.E(real_images)
        x_recon = self.G(z_encoded_ge)
        _, feats_real = self.D(real_images, z_encoded_ge)
        _, feats_recon = self.D(x_recon, z_encoded_ge)
        recon_loss = nn.L1Loss(reduction="sum")(feats_recon, feats_real)
        recon_loss = recon_loss * 5e-5 # Scaled to balance with BCEs

        # Combined G+E loss and scaled reconstruction loss
        ge_loss = g_loss + e_loss + recon_loss

        # Backward + Step
        ge_loss.backward()
        self.G_optim.step()
        self.E_optim.step()

        return d_loss.item(), g_loss.item() + e_loss.item(), recon_loss.item()

    def validate_step(self, real_images):
        with torch.no_grad():
            real_images = real_images.to(self.device)
            batch_size = real_images.size(0)
            
            z_prior = torch.randn(batch_size, self.latent_dim, device=self.device)
            
            z_encoded = self.E(real_images)
            x_generated = self.G(z_prior)
            DE, _ = self.D(real_images, z_encoded)
            DG, _ = self.D(x_generated, z_prior)

            d_loss = self.D_loss(DG, DE)
            
            g_loss, e_loss = self.GE_losses(DG, DE)

            _, feats_real = self.D(real_images, z_encoded)
            _, feats_recon = self.D(self.G(z_encoded), z_encoded)
            recon_loss = nn.L1Loss(reduction="sum")(feats_recon, feats_real)
            recon_loss = recon_loss * 5e-5

            return d_loss.item(), g_loss.item() + e_loss.item(), recon_loss.item()

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
    
    # Test EGBAD
    egbad = EGBAD(filter_size=32, latent_dim=128, device=device)
    test_batch = torch.randn(4, 1, 256, 256).to(device)
    d_loss, ge_loss, recon_loss = egbad.train_step(test_batch)
    print(f"Training step - D loss: {d_loss:.4f}, GE loss: {ge_loss:.4f}, Recon loss: {recon_loss:.4f}")
    generated_samples = egbad.generate_samples(num_samples=4)
    print("Generated samples shape:", generated_samples.shape)

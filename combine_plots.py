import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch

# Resim dosyalarının yolları
images = [
    "chebyshev_g0.png",  # Sol üst
    "jacobi_g0.png",  # Sağ üst
    "legendre_g0.png",  # Sol alt
    "monomial_g0.png"   # Sağ alt
]

# Her resim için başlıklar
titles = [
    "Chebyshev",
    "Jacobi",
    "Legendre",
    "Monomial"
]

# Ana başlık
main_title = "Comparison of the actual and predicted traffic flow (G_1 filter)"

# Figure oluştur - A4 boyutuna yakın
fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))  # A4 landscape boyutu
fig.suptitle(main_title, fontsize=16, fontweight='bold', y=0.98)

# Her bir subplot için resimleri yerleştir
for idx, (ax, img_path, title) in enumerate(zip(axes.flat, images, titles)):
    try:
        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.axis('off')
        
        # Başlığı resmin altına ekle
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    except FileNotFoundError:
        ax.text(0.5, 0.5, f'Image not found:\n{img_path}', 
                ha='center', va='center', fontsize=10, color='red')
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
        ax.axis('off')

# Layout ayarları
plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.96])  # Ana başlık için yer bırak

# Dış çerçeve ekle (opsiyonel)
fig.patch.set_edgecolor('navy')
fig.patch.set_linewidth(3)

# Kaydet
plt.savefig('combined_traffic_plots.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='navy', pad_inches=0.1)
print("✅ Birleştirilmiş grafik 'combined_traffic_plots.png' olarak kaydedildi!")

plt.show()

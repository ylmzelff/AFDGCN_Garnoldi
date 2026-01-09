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

# Figure oluştur - Optimize edilmiş boyut
fig, axes = plt.subplots(2, 2, figsize=(12, 9.5))
fig.suptitle(main_title, fontsize=18, fontweight='bold', y=0.985)

# Subplot'lar arası boşluğu minimize et
plt.subplots_adjust(hspace=0.08, wspace=0.08, left=0.03, right=0.97, top=0.87, bottom=0.02)

# Her bir subplot için resimleri yerleştir
for idx, (ax, img_path, title) in enumerate(zip(axes.flat, images, titles)):
    try:
        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.axis('off')
        
        # Başlığı resmin altına ekle - daha kompakt
        ax.set_title(title, fontsize=15, fontweight='bold', pad=3)
    except FileNotFoundError:
        ax.text(0.5, 0.5, f'Image not found:\n{img_path}', 
                ha='center', va='center', fontsize=10, color='red')
        ax.set_title(title, fontsize=15, fontweight='bold', pad=3)
        ax.axis('off')

# Ortak legend oluştur - state of the art style
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

legend_elements = [
    Line2D([0], [0], color='blue', linewidth=2, label='Real Traffic Flow'),
    Line2D([0], [0], color='orange', linewidth=2, linestyle='--', label='Predicted Traffic Flow'),
    Patch(facecolor='yellow', alpha=0.3, label='Zoomed Region')
]

# Legend'ı ana başlığın altına, grafiklerin üstüne yerleştir
fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.92),
          ncol=3, frameon=True, fontsize=11, framealpha=0.95, edgecolor='gray', fancybox=True)

# Layout ayarları - manuel ayarlarla optimize edildi, tight_layout kullanmıyoruz
# plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.96])

# Dış çerçeve - daha ince ve profesyonel
fig.patch.set_edgecolor('navy')
fig.patch.set_linewidth(2)

# Kaydet
plt.savefig('combined_traffic_plots.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='navy', pad_inches=0.1)
print("✅ Birleştirilmiş grafik 'combined_traffic_plots.png' olarak kaydedildi!")

plt.show()

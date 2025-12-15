import streamlit as st
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

st.set_page_config(
    page_title="Dashboard Analisis Resep Obat",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# UPLOAD FILE
# ============================================

if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

uploaded_placeholder = st.empty()

if st.session_state.uploaded_file is None:
    with uploaded_placeholder:
        uploaded_file = st.file_uploader(
            "📤 Unggah file CSV data resep obat", type=["csv"]
        )
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        uploaded_placeholder.empty()
        st.rerun()
    else:
        st.title("Sistem Rekomendasi Resep Obat")
        st.info("Silakan unggah file CSV untuk memulai analisis.")
        st.stop()



# ============================================
# LOAD DATA
# ============================================
uploaded_file = st.session_state.uploaded_file
uploaded_file.seek(0)
df = pd.read_csv(uploaded_file)

# ============================================
# PREPROCESSING
# ============================================
df = df.dropna(subset=['Aturan Pakai']).drop_duplicates()

# Normalisasi teks
for col in ['Nama Obat', 'Aturan Pakai', 'Nama Pasien', 'Nama Dokter', 'Ruangan', 'Status']:
    if col in df.columns:
        df[col] = df[col].astype(str).str.lower().str.strip()

# Membuat ID Resep unik
df["Id Resep"] = df["Tanggal"].astype(str).str.strip() + " " + df["Waktu"].astype(str).str.strip()

# Konversi waktu ke jam
try:
    df["Waktu"] = pd.to_datetime(df["Waktu"], format="%H:%M:%S").dt.hour
except:
    pass

# Kategori waktu
def kategori_waktu(waktu):
    jam = waktu.hour
    if 0 <= jam < 12:
        return "Pagi"
    elif 12 <= jam < 17:
        return "Siang"
    elif 17 <= jam < 21:
        return "Sore"
    else:
        return "Malam"

if "Waktu" in df.columns:
    df["Kategori Waktu"] = df["Waktu"].apply(kategori_waktu)

# Konversi tanggal & hari
df["Tanggal"] = pd.to_datetime(df["Tanggal"], errors="coerce")
df = df.dropna(subset=["Tanggal"])
df["Hari"] = df["Tanggal"].dt.day_name()

# ============================================
# PENANGANAN OUTLIER (JUMLAH OBAT PER RESEP)
# ============================================

# Hitung jumlah obat per transaksi
jumlah_obat = df.groupby(["Tanggal", "Waktu", "Id Resep"])["Nama Obat"].count()

# Hitung IQR
Q1 = jumlah_obat.quantile(0.25)
Q3 = jumlah_obat.quantile(0.75)
IQR = Q3 - Q1

# Tentukan batas outlier
batas_bawah = Q1 - 1.5 * IQR
batas_atas = Q3 + 1.5 * IQR

# Identifikasi transaksi normal
transaksi_normal = jumlah_obat[
    (jumlah_obat >= batas_bawah) & (jumlah_obat <= batas_atas)
]

# Ambil index transaksi normal
index_normal = transaksi_normal.index

# Filter dataframe utama (silent)
df = (
    df.set_index(["Tanggal", "Waktu", "Id Resep"])
      .loc[index_normal]
      .reset_index()
)

# ============================================
# SIDEBAR MENU
# ============================================
st.sidebar.title("📌 Menu")
menu = st.sidebar.radio(
    "Pilih halaman:",
    ["Dashboard", "Distribusi Obat", "Analisis Kombinasi Obat"]
)

# ============================================
# DASHBOARD
# ============================================
if menu == "Dashboard":
    st.title("📊 Dashboard Analisis Resep Obat")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Jumlah Resep", len(df["Id Resep"].unique()))
    col2.metric("Jenis Obat", df["Nama Obat"].nunique())
    col3.metric("Jumlah Dokter", df["Nama Dokter"].nunique() if "Nama Dokter" in df.columns else "-")
    col4.metric("Rentang Tanggal", f"{df['Tanggal'].min()} – {df['Tanggal'].max()}")

    st.markdown("---")

    colA, colB = st.columns(2)

    with colA:
        st.subheader("🔝 Top 10 Obat Paling Banyak Diresepkan")
        top_10 = df["Nama Obat"].value_counts().head(10)
        st.bar_chart(top_10)

    with colB:
        st.subheader("🔻 Top 10 Obat Paling Jarang Diresepkan")
        rare_10 = df["Nama Obat"].value_counts().sort_values().head(10)
        st.bar_chart(rare_10)

    st.markdown("---")

    colC, colD = st.columns(2)

    with colC:
        st.subheader("⏰ Grafik Transaksi Berdasarkan Waktu (Pagi–Malam)")
        st.info("""
        **Pembagian Waktu Transaksi Obat:**
        - **Pagi:** 00:00 – 11:59  
        - **Siang:** 12:00 – 16:59  
        - **Sore:** 17:00 – 21:00  
        - **Malam:** 21:00 – 23:59  
        """)
        waktu_kat_count = df["Kategori Waktu"].value_counts()
        kategori_lengkap = ["Pagi", "Siang", "Sore", "Malam"]
        waktu_kat_count = waktu_kat_count.reindex(kategori_lengkap, fill_value=0)
        st.bar_chart(waktu_kat_count)

    with colD:
        st.subheader("📅 Grafik Transaksi Berdasarkan Hari")
        order_hari = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        hari_count = df["Hari"].value_counts().reindex(order_hari, fill_value=0)
        grafik_hari_df = pd.DataFrame({
            "Hari": order_hari,
            "Total Transaksi": hari_count.values
        }).set_index("Hari")
        st.bar_chart(grafik_hari_df)


# ============================================
# DISTRIBUSI OBAT
# ============================================
elif menu == "Distribusi Obat":
    st.title("📦 Analisis Distribusi Obat")

    col1, col2 = st.columns(2)
    unique_dates = sorted(df["Tanggal"].astype(str).unique())
    date_filter = col1.multiselect("📅 Filter tanggal", unique_dates)

    df_filtered = df[df["Tanggal"].astype(str).isin(date_filter)] if date_filter else df

    if "Nama Dokter" in df.columns:
        dokter_filter = col2.selectbox("👨‍⚕️ Filter dokter", ["Semua"] + df["Nama Dokter"].unique().tolist())
        if dokter_filter != "Semua":
            df_filtered = df_filtered[df_filtered["Nama Dokter"] == dokter_filter]

    st.success(f"Jumlah data setelah filter: {len(df_filtered)}")

    st.subheader("📋 Tabel Distribusi Obat")
    tabel_distribusi = (
        df_filtered["Nama Obat"]
        .value_counts()
        .reset_index()
        .rename(columns={"index": "Nama Obat", "Nama Obat": "Total Resep"})
    )
    st.dataframe(tabel_distribusi, use_container_width=True)


# ============================================
# ANALISIS KOMBINASI OBAT
# ============================================
elif menu == "Analisis Kombinasi Obat":
    st.title("🧠 Analisis Pola Kombinasi Obat")

    st.sidebar.header("⚙️ Parameter ")
    min_support = st.sidebar.slider("Support minimum (%)", 1, 20, 5) / 100
    min_confidence = st.sidebar.slider("Confidence minimum (%)", 10, 100, 50) / 100
    top_n = st.sidebar.slider("Jumlah aturan ditampilkan", 5, 30, 10)

    # transaksi obat
    transaksi = df.groupby("Id Resep")["Nama Obat"].apply(list).tolist()

    te = TransactionEncoder()
    te_array = te.fit(transaksi).transform(transaksi)
    df_te = pd.DataFrame(te_array, columns=te.columns_)

    frequent_items = fpgrowth(df_te, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_items, metric="confidence", min_threshold=min_confidence)

    if rules.empty:
        st.warning("Tidak ditemukan kombinasi obat dengan parameter ini.")
        st.stop()

    # Normalisasi aturan
    rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
    rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))
    rules_sorted = rules.sort_values(by="lift", ascending=False)

    # tampilkan aturan
    st.subheader("📌 Aturan Kombinasi Obat")


    st.markdown("### 📋 Tabel Aturan Kombinasi")
    st.dataframe(rules_sorted.head(top_n), use_container_width=True)

    # =================================================
    # REKOMENDASI STOK OBAT
    # =================================================
    
    st.subheader("📦 Rekomendasi Stok Obat")

    stok_freq = {}

    for _, row in rules_sorted.iterrows():
        for ob in row["antecedents"].split(", "):
            stok_freq[ob] = stok_freq.get(ob, 0) + 1
        for ob in row["consequents"].split(", "):
            stok_freq[ob] = stok_freq.get(ob, 0) + 1

    stok_df = (
    pd.DataFrame(
        list(stok_freq.items()),
        columns=["Nama Obat", "Frekuensi Muncul di Aturan"]
    )
    .sort_values("Frekuensi Muncul di Aturan", ascending=False)
    .reset_index(drop=True)
    )
    
    threshold = stok_df["Frekuensi Muncul di Aturan"].max() * 0.7
    
    stok_df["Rekomendasi Stok"] = stok_df["Frekuensi Muncul di Aturan"].apply(
    lambda x: "⚠️ Tinggi" if x >= threshold else "Normal"
    )

    st.dataframe(stok_df, use_container_width=True)

    # =================================================
    # REKOMENDASI PENEMPATAN OBAT
    # =================================================
    st.subheader("📍 Rekomendasi Penempatan Obat")

    penempatan_list = []

    for _, row in rules_sorted.head(15).iterrows():
        penempatan_list.append({
            "Obat 1": row["antecedents"],
            "Obat 2": row["consequents"],
            "Lift": row["lift"],
            "Rekomendasi": f"Tempatkan **{row['antecedents']}** dekat dengan **{row['consequents']}**"
        })

    penempatan_df = pd.DataFrame(penempatan_list)
    st.dataframe(penempatan_df, use_container_width=True)

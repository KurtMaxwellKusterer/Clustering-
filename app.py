import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

from clustering import load_data, run_kmeans, compute_wcss, run_hierarchical

st.set_page_config(
    page_title='Customer Segmentation Dashboard',
    layout='wide',
    initial_sidebar_state='expanded'
)

# ── Caching ───────────────────────────────────────────────────────────────────

@st.cache_data
def get_data():
    return load_data()

@st.cache_data
def get_clusters(k):
    return run_kmeans(load_data(), k)

@st.cache_data
def get_wcss():
    return compute_wcss(load_data())

@st.cache_data
def get_hierarchical():
    return run_hierarchical(load_data())

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title('Settings')
    k = st.slider('Number of Clusters (k)', min_value=2, max_value=10, value=5)
    st.caption(
        'Adjust k to explore different segmentations. '
        'Check the Elbow chart on the Segmentation tab to find the optimal value.'
    )
    st.divider()
    st.markdown('**Dataset**')
    df_raw = get_data()
    st.caption(f'{len(df_raw)} customers · 4 features')

# ── Data ──────────────────────────────────────────────────────────────────────

df, segment_map, silhouette, centroids = get_clusters(k)
wcss = get_wcss()

COLOURS = px.colors.qualitative.Vivid

# ── Header ────────────────────────────────────────────────────────────────────

st.title('Customer Buying Behaviour Dashboard')
st.markdown('K-Means and Hierarchical Clustering analysis of mall customer spending patterns.')
st.divider()

# ── KPI Row ───────────────────────────────────────────────────────────────────

c1, c2, c3, c4 = st.columns(4)
c1.metric('Total Customers', len(df))
c2.metric('Avg Annual Income', f"${df['Annual Income (k$)'].mean():.0f}k")
c3.metric('Avg Spending Score', f"{df['Spending Score (1-100)'].mean():.0f} / 100")
c4.metric(
    'Silhouette Score', f"{silhouette:.2f}",
    help='Ranges from -1 to 1. Closer to 1 means well-separated clusters.'
)

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(['Overview', 'Segmentation', 'Hierarchical'])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════

with tab1:
    st.subheader('Customer Demographics & Spending Patterns')

    # Age groups
    df_plot = df.copy()
    age_bins = [18, 26, 36, 46, 56, 66, 100]
    age_labels = ['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
    df_plot['Age Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels)
    age_counts = df_plot.groupby('Age Group', observed=True).size().reset_index(name='Count')

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            age_counts, x='Age Group', y='Count',
            title='Customer Distribution by Age Group',
            color='Count', color_continuous_scale='Viridis',
            text='Count'
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(coloraxis_showscale=False, yaxis_title='Number of Customers')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(
            df, x='Genre', y='Spending Score (1-100)',
            title='Spending Score by Gender',
            color='Genre',
            color_discrete_sequence=[COLOURS[0], COLOURS[2]],
            points='all'
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        fig = px.histogram(
            df, x='Spending Score (1-100)', nbins=20,
            title='Spending Score Distribution',
            color_discrete_sequence=[COLOURS[3]],
            marginal='box'
        )
        fig.update_layout(bargap=0.05, yaxis_title='Number of Customers')
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        avg_income = (
            df_plot.groupby('Age Group', observed=True)['Annual Income (k$)']
            .mean().round(1).reset_index()
        )
        fig = px.bar(
            avg_income, x='Age Group', y='Annual Income (k$)',
            title='Average Annual Income by Age Group',
            color='Annual Income (k$)', color_continuous_scale='Purples',
            text='Annual Income (k$)'
        )
        fig.update_traces(textposition='outside', texttemplate='$%{text}k')
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — SEGMENTATION
# ═════════════════════════════════════════════════════════════════════════════

with tab2:
    st.subheader(f'K-Means Segmentation  —  k = {k}')

    col1, col2 = st.columns([3, 2])

    with col1:
        fig = px.scatter(
            df,
            x='Annual Income (k$)',
            y='Spending Score (1-100)',
            color='Segment',
            symbol='Genre',
            hover_data={'Age': True, 'Genre': True, 'Cluster': False},
            title='Customer Segments by Income & Spending Score',
            color_discrete_sequence=COLOURS,
            size_max=10,
            opacity=0.8
        )
        # Annotate centroids
        for i, (income, spend) in enumerate(centroids):
            fig.add_annotation(
                x=income, y=spend,
                text=f'<b>{segment_map[i]}</b>',
                showarrow=False,
                font=dict(size=10, color='black'),
                bgcolor='rgba(255,255,255,0.95)',
                bordercolor='#333333',
                borderwidth=1.5,
                opacity=1.0
            )
        fig.update_layout(legend=dict(orientation='h', yanchor='top', y=-0.2))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        elbow_df = pd.DataFrame({'k': range(1, 11), 'WCSS': wcss})
        fig = px.line(
            elbow_df, x='k', y='WCSS',
            markers=True,
            title='Elbow Method — Optimal k',
            labels={'k': 'Number of Clusters', 'WCSS': 'Within-Cluster Sum of Squares'}
        )
        fig.add_vline(x=k, line_dash='dash', line_color='red',
                      annotation_text=f'  k={k} selected', annotation_position='top right')
        st.plotly_chart(fig, use_container_width=True)

        # Segment size donut
        size_df = df['Segment'].value_counts().reset_index()
        size_df.columns = ['Segment', 'Count']
        fig = px.pie(
            size_df, names='Segment', values='Count',
            title='Customers per Segment',
            color_discrete_sequence=COLOURS,
            hole=0.45
        )
        fig.update_traces(textposition='inside', textinfo='percent+label',
                          textfont=dict(color='white', size=12))
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader('Segment Feature Profiles')

    col3, col4 = st.columns(2)

    profile = (
        df.groupby('Segment')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
        .mean().round(1).reset_index()
    )

    with col3:
        profile_melted = profile.melt(
            id_vars='Segment', var_name='Feature', value_name='Mean Value'
        )
        fig = px.bar(
            profile_melted, x='Segment', y='Mean Value',
            color='Segment', facet_col='Feature',
            title='Mean Feature Values per Segment',
            color_discrete_sequence=COLOURS,
            text='Mean Value'
        )
        fig.update_traces(textposition='outside', texttemplate='%{text:.1f}')
        fig.update_xaxes(showticklabels=False, title='')
        fig.update_yaxes(matches=None)
        fig.for_each_annotation(lambda a: a.update(text=a.text.split('=')[-1]))
        fig.update_layout(showlegend=True,
                          legend=dict(orientation='h', yanchor='bottom', y=-0.25))
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        corr_features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
        segment_dummies = pd.get_dummies(df['Segment'])
        corr_matrix = pd.DataFrame(
            {seg: [df[feat].corr(segment_dummies[seg]) for feat in corr_features]
             for seg in segment_dummies.columns},
            index=corr_features
        ).T

        fig = px.imshow(
            corr_matrix,
            text_auto='.2f',
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1,
            title='Segment–Feature Pearson Correlation',
            labels=dict(x='Feature', y='Segment', color='Correlation'),
            aspect='auto'
        )
        fig.update_traces(textfont=dict(color='black', size=13))
        fig.update_layout(
            xaxis_title='',
            yaxis_title='',
            coloraxis_colorbar=dict(
                title='Correlation',
                tickvals=[-1, -0.5, 0, 0.5, 1],
                ticktext=['-1.0', '-0.5', '0', '0.5', '1.0']
            )
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader('Segment Summary Table')
    summary = (
        df.groupby('Segment')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
        .agg(['mean', 'min', 'max']).round(1)
    )
    summary.columns = [' '.join(c).title() for c in summary.columns]
    summary['Customer Count'] = df.groupby('Segment').size()
    st.dataframe(summary, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — HIERARCHICAL
# ═════════════════════════════════════════════════════════════════════════════

with tab3:
    st.subheader('Hierarchical Clustering')

    Z, features_scaled = get_hierarchical()

    st.markdown(
        'The dendrogram shows how customers are progressively merged into groups by similarity. '
        'Branches that join lower on the y-axis indicate more similar customers. '
        'The dashed red line shows where cutting the tree produces clusters comparable to K-Means.'
    )

    # Dendrogram via plotly figure_factory
    fig = ff.create_dendrogram(
        features_scaled.values,
        orientation='bottom',
        color_threshold=0.7
    )
    fig.update_layout(
        title='Customer Dendrogram (Ward Linkage)',
        xaxis=dict(title='Customers (individual IDs suppressed)', showticklabels=False),
        yaxis=dict(title='Dissimilarity'),
        height=500
    )
    # Cut line
    fig.add_hline(y=0.7, line_dash='dash', line_color='red',
                  annotation_text='Cut threshold', annotation_position='top right')
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader('Feature Heatmap by K-Means Segment')
    st.markdown(
        'Customers are sorted by their K-Means segment. '
        'Darker cells indicate higher normalised values for that feature.'
    )

    # Sort customers by segment for readable banding
    df_sorted = df.copy()
    df_sorted['_sort'] = df_sorted['Segment'].map(
        {s: i for i, s in enumerate(sorted(df['Segment'].unique()))}
    )
    df_sorted = df_sorted.sort_values('_sort')

    heatmap_features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    from sklearn.preprocessing import MinMaxScaler
    scaler_h = MinMaxScaler()
    heatmap_data = pd.DataFrame(
        scaler_h.fit_transform(df_sorted[heatmap_features]),
        columns=heatmap_features
    ).T

    fig = px.imshow(
        heatmap_data,
        color_continuous_scale='YlOrRd',
        title='Normalised Customer Feature Heatmap (sorted by segment)',
        labels=dict(x='Customer', y='Feature', color='Normalised Value'),
        aspect='auto'
    )
    fig.update_xaxes(showticklabels=False)
    st.plotly_chart(fig, use_container_width=True)

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from bernoulli_solver import bernoulli_equation, secant_method, analytical_solution

st.set_page_config(page_title="Solusi Numerik Persamaan Bernoulli", layout="wide")

def create_equation_section():
    st.header("Metode Secant untuk Persamaan Bernoulli")
    st.markdown("""
    <style>
    .equation {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 20px 0;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="equation">
    <h3>Persamaan Bernoulli:</h3>
    </div>
    """, unsafe_allow_html=True)

    st.latex(r"P_1 + \frac{1}{2}\rho v_1^2 + \rho gh_1 = P_2 + \frac{1}{2}\rho v_2^2 + \rho gh_2")

    st.markdown("""
    Menyelesaikan Persamaan Bernoulli untuk mencari kecepatan fluida pada titik 2 ($v_2$).
    """)

def validate_inputs(P1, P2, rho, g, h1, h2, v1):
    if rho <= 0:
        return False, "Nilai Densitas Harus Positif"
    if v1 < 0:
        return False, "Kecepatan Harus Positif (non-negative)"
    if P1 < 0 or P2 < 0:
        return False, "Tekanan harus Positif"
    return True, ""

def suggest_initial_guesses(v1, P1, P2, rho, g, h1, h2):
    try:
        v2_analytical = analytical_solution(P1, P2, rho, g, h1, h2, v1)
        return max(0.1, v2_analytical * 0.5), min(v2_analytical * 1.5, 20.0)
    except:
        # If analytical solution fails, use v1-based guesses
        return max(0.1, v1 * 0.5), min(v1 * 1.5, 20.0)

def create_input_section():
    with st.container():
        st.subheader("Parameter Sistem")

        # Fluid properties
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Sifat Fluida")
            rho = st.number_input(
                "Densitas Fluida (ρ) [kg/m³]",
                value=1000.0,
                min_value=0.1,
                help="Massa per satuan volume fluida. [kg/m³]\nContoh:\n- Air: 1000 kg/m³\n- Udara: 1.225 kg/m³",
                format="%.1f"
            )
        with col2:
            g = 9.81  # gravitational acceleration
            st.markdown("""
            **Percepatan Gravitasi (g)**: 9.81 m/s²
            """)

        # Point 1 and 2 parameters
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Parameter pada titik 1")
            P1 = st.number_input(
                "Tekanan pada titik 1 (P₁) [Pa]",
                value=101325.0,
                help="Tekanan pada titik 1 dalam Pascals [Pa]\nContoh:\n- Tekanan Atmosfer: 101325 Pa\n- 1 bar = 100000 Pa",
                format="%.1f"
            )
            h1 = st.number_input(
                "Ketinggian pada titik 1 (h₁) [m]",
                value=0.0,
                help="Ketinggian pada titik 1 dalam meter [m]",
                format="%.2f"
            )
            v1 = st.number_input(
                "Velocity at Point 1 (v₁) [m/s]",
                value=2.0,
                min_value=0.0,
                help="Kecepatan fluida pada titik 1 dalam mps [m/s]\contoh: 0.1 - 20 m/s untuk aliran air",
                format="%.2f"
            )

        with col2:
            st.markdown("#### Parameter pada titik 2")
            P2 = st.number_input(
                "Tekanan pada titik 2 (P₂) [Pa]",
                value=101325.0,
                help="Tekanan pada titik 2 [Pa]\nContoh:\n- Tekanan atmosfer: 101325 Pa\n- 1 bar = 100000 Pa",
                format="%.1f"
            )
            h2 = st.number_input(
                "Ketinggian pada titik 2 (h₂) [m]",
                value=1.0,
                help="Ketinggian pada titik 2 [m]",
                format="%.2f"
            )

    # Add suggested guesses
    suggested_x0, suggested_x1 = suggest_initial_guesses(v1, P1, P2, rho, g, h1, h2)

    # Numerical method parameters in an expander
    with st.expander("Parameter Metode Secant", expanded=False):
        st.markdown("""
      ### Metode Secant untuk v₂  
      Metode secant memerlukan dua tebakan awal untuk v₂ [m/s]. Tebakan yang baik membantu nilai error konvergen lebih cepat.  

    #### **Saran secara hukum fisika:**  
    - Berdasarkan hukum kekekalan energi, v₂ seharusnya mirip dengan v₁.  
    - Untuk pipa yang menyempit, v₂ > v₁.  
    - Untuk pipa yang melebar, v₂ < v₁.  

    #### **Rentang yang direkomendasikan:**  
    - Mulailah dengan nilai yang mendekati v₁.  
    - Pastikan perkiraan memiliki jarak yang cukup masuk akal.  

        """)

        col1, col2 = st.columns(2)
        with col1:
            x0 = st.number_input(
                "Tebakan 1 [m/s]",
                value=suggested_x0,
                min_value=0.1,
                help="Tebakan 1 untuk nilai v₂ dalam meters per second [m/s]",
                format="%.2f"
            )
        with col2:
            x1 = st.number_input(
                "Tebakan 2 [m/s]",
                value=suggested_x1,
                min_value=0.1,
                help="Tebakan 2 untuk nilai v₂ dalam meters per second [m/s]",
                format="%.2f"
            )

    return P1, P2, rho, g, h1, h2, v1, x0, x1

def plot_convergence(iterations):
    fig, ax = plt.subplots(figsize=(10, 6))
    iterations_array = np.array(iterations)

    ax.semilogy([i[0] for i in iterations], 
                [abs(i[2]) for i in iterations], 
                'b.-', linewidth=2, markersize=8)

    ax.set_xlabel('Iteration Number')
    ax.set_ylabel('Absolute Error |f(x)|')
    ax.set_title('Convergence History')
    ax.grid(True, which="both", ls="-", alpha=0.2)

    # Add points and annotations
    for i, (iter_num, x_val, fx_val) in enumerate(iterations):
        ax.annotate(f'v₂ = {x_val:.2f}',
                   (iter_num, abs(fx_val)),
                   textcoords="offset points",
                   xytext=(0,10),
                   ha='center')

    return fig

def plot_flow_visualization(h1, h2, v1, v2):
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot pipe
    pipe_width = 1.0
    ax.fill_between([0, 5], [h1-pipe_width/2, h2-pipe_width/2], 
                   [h1+pipe_width/2, h2+pipe_width/2], 
                   color='lightgray', alpha=0.3)

    # Plot flow arrows
    x1, x2 = 1, 4
    arrow_density = 20
    x_points = np.linspace(x1, x2, arrow_density)

    for x in x_points:
        y = h1 + (h2-h1)*(x-x1)/(x2-x1)
        dx = 0.2
        dy = 0.1

        # Scale arrow size with velocity
        v_local = v1 + (v2-v1)*(x-x1)/(x2-x1)
        scale = v_local / max(v1, v2)

        ax.arrow(x, y, dx*scale, dy*(h2-h1)/(x2-x1), 
                head_width=0.1, head_length=0.1, fc='blue', ec='blue', alpha=0.6)

    # Add velocity labels
    ax.text(x1, h1+0.5, f'v₁ = {v1:.2f} m/s', ha='center')
    ax.text(x2, h2+0.5, f'v₂ = {v2:.2f} m/s', ha='center')

    # Add height labels
    ax.text(0.5, h1, f'h₁ = {h1:.2f} m', va='center')
    ax.text(4.5, h2, f'h₂ = {h2:.2f} m', va='center')

    ax.set_xlim(0, 5)
    ax.set_ylim(min(h1,h2)-1.5, max(h1,h2)+1.5)
    ax.set_xlabel('Distance')
    ax.set_ylabel('Height (m)')
    ax.set_title('Flow Visualization')
    ax.grid(True, alpha=0.2)

    return fig

def main():
    create_equation_section()
    P1, P2, rho, g, h1, h2, v1, x0, x1 = create_input_section()

    # Solve button with custom styling
    solve_button = st.button("Solve for v₂", type="primary")

    if solve_button:
        # Validate inputs first
        valid, message = validate_inputs(P1, P2, rho, g, h1, h2, v1)
        if not valid:
            st.error(f"Invalid input parameters: {message}")
            return

        try:
            # Try analytical solution first
            v2_analytical = analytical_solution(P1, P2, rho, g, h1, h2, v1)

            # Calculate numerical solution
            params = (P1, P2, rho, g, h1, h2, v1)
            v2_numerical, iterations, converged = secant_method(bernoulli_equation, x0, x1, params)

            if converged and v2_numerical is not None:
                st.success(f"""
                ✨ Solusi didapatkan:
                - Analitik: v₂ = {v2_analytical:.3f} m/s
                - Numerik (Metode Secant): v₂ = {v2_numerical:.3f} m/s
                - Error Relatif: {abs(v2_analytical - v2_numerical)/v2_analytical*100:.6f}%
                """)

                # Create tabs for different visualizations
                tab1, tab2, tab3 = st.tabs(["Visualisasi Aliran Fluida", "Grafik Error (konvergensi)", "Detail Iterasi"])

                with tab1:
                    st.pyplot(plot_flow_visualization(h1, h2, v1, v2_numerical))

                with tab2:
                    st.pyplot(plot_convergence(iterations))

                with tab3:
                    st.markdown("#### Iteration History")
                    iteration_data = []
                    for i, x, fx in iterations:
                        iteration_data.append({
                            "Iteration": i,
                            "v₂ Value (m/s)": f"{x:.6f}",
                            "Error": f"{abs(fx):.2e}"
                        })
                    st.table(iteration_data)
            else:
                st.error("""
            ### Metode Numerik Tidak Mencapai Konvergensi  

            Hal ini mungkin disebabkan oleh:  
            1. Tebakan awal terlalu jauh dari solusi.  
            2. Masalah yang dianalisis mungkin tidak realistis secara fisika.  
            3. Ketidakstabilan numerik dalam perhitungan.  

            #### **Coba langkah berikut:**  
            1. Gunakan tebakan awal yang disarankan.  
            2. Pastikan parameter input yang digunakan masuk akal secara fisika.  
            3. Jika Anda mengetahui perkiraan jawabannya, gunakan tebakan di sekitar nilai tersebut.  

                """)

                # Show analytical solution anyway if available
                st.info(f"Rekomendasi nilai tebakan untuk v₂ ≈ {v2_analytical:.3f} m/s")

        except ValueError as e:
            st.error(f"Error in calculation: {str(e)}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()

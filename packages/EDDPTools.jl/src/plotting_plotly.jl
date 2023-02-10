using EDDP
using EDDP: ComputedRecord, PhaseDiagram, get_e_above_hull
using CellBase
using LaTeXStrings
import PlotlyJS

export make_binary_hull_plotly, make_ternary_plot


"""
    make_ternary_plot(plot_data)

Compose ternary diagram from a PhaseDiagram.
"""
function make_ternary_plot(phased::PhaseDiagram)
    make_ternary_plot(EDDP.get_ternary_hulldata(phased))
end

function make_ternary_plot(plot_data)

    stable_mask = plot_data.stable_mask
    unstable_mask = plot_data.unstable_mask
    abc_stable = plot_data.abc[:, stable_mask]
    abc_unstable = plot_data.abc[:, unstable_mask]
    reduced_formula = plot_data.reduced_formula
    formation_energies = plot_data.formation_energies
    labels = plot_data.labels
    elements = plot_data.elements
    e_above_hull = plot_data.e_above_hull

    function make_ax(title, tickangle)
        PlotlyJS.attr(
            title=title,
            titlefont_size=20,
            tickangle=tickangle,
            tickfont_size=15,
            tickcolor="rgba(0, 0, 0, 0)",
            ticklen=5,
            showline=true,
            showgrid=true,
        )
    end

    has_unstable = sum(unstable_mask) > 0
    t_stable = PlotlyJS.scatterternary(
        name="Stable",
        mode="markers",
        a=abc_stable[1, :],
        b=abc_stable[2, :],
        c=abc_stable[3, :],
        customdata=collect(
            zip(
                reduced_formula[stable_mask],
                labels[stable_mask],
                formation_energies[stable_mask],
            ),
        ),
        marker=PlotlyJS.attr(symbol="diamond", color="#60dbf1", size=14, line_width=2),
        hovertemplate="Composition: %{customdata[0]}<br>Label: %{customdata[1]}<br>Formation energy: %{customdata[2]:.5f} eV/atom<br><extra></extra>",
    )
    traces = [t_stable]
    if has_unstable
        t_unstable = PlotlyJS.scatterternary(
            name="Unstable",
            mode="markers",
            a=abc_unstable[1, :],
            b=abc_unstable[2, :],
            c=abc_unstable[3, :],
            customdata=collect(
                zip(
                    reduced_formula[unstable_mask],
                    labels[unstable_mask],
                    e_above_hull[unstable_mask],
                    formation_energies[unstable_mask],
                ),
            ),
            hovertemplate="Composition: %{customdata[0]}<br>Label: %{customdata[1]}<br>e_above_hull: %{customdata[2]:.5f} eV/atom<br>Formation energy: %{customdata[3]:.5f} eV/atom<extra></extra>",
            marker=PlotlyJS.attr(symbol="circle", color="#DB7365", size=8, line_width=0),
        )
        push!(traces, t_unstable)
    end



    layout = PlotlyJS.Layout(
        ternary=PlotlyJS.attr(
            sum=1,
            aaxis=make_ax("$(elements[1])", 0),
            baxis=make_ax("$(elements[2])", 45),
            caxis=make_ax("$(elements[3])", -45),
            bgcolor="#f5f5f5",
        ),
        paper_bgcolor=" #f5f5f5",
    )
    PlotlyJS.plot(traces, layout)
end


"""
    make_binary_hull_plotly(ps::EDDP.PhaseDiagram)

Generate binary convex hull plot.
"""
make_binary_hull_plotly(ps::EDDP.PhaseDiagram) =
    make_binary_hull_plotly(EDDP.get_2d_plot_data(ps))

"""
    make_binary_hull_plotly(data)

Generate binary convex hull plot.

- `data`: A `NamedTuple` return by `EDDP.get_2d_plot_data`.
"""
function make_binary_hull_plotly(data)
    trace1 = PlotlyJS.scatter(
        mode="markers",
        x=data.x,
        y=data.y,
        customdata=[zip(data.e_above_hull, data.record_ids, data.record_formula)...],
        hovertemplate="Unstable Candidate<br>Distance to hull: %{customdata[0]:.5f} eV<br>Record-id: %{customdata[1]}<br>Composition: %{customdata[2]}",
        name="Candidate",
    )
    trace2 = PlotlyJS.scatter(
        mode="lines+markers",
        name="Hull",
        x=data.stable_x,
        y=data.stable_y,
        customdata=[zip(data.stable_formula, data.stable_entry_id)...],
        hovertemplate="Stable Candidate<br>Formula: %{customdata[0]}<br>Record-id: %{customdata[1]}",
    )
    comp_label = "$(data.elements[2])x$(data.elements[1])1-x"
    layout = PlotlyJS.Layout(
        title="Binary Hull of $(data.elements[1])-$(data.elements[2])",
        xaxis_title=comp_label,
        yaxis_title="Formation energy (eV / atom)",
    )
    PlotlyJS.plot([trace1, trace2], layout)
end

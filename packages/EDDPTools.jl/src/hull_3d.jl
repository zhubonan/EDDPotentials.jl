using EDDP
using EDDP: ComputedRecord, PhaseDiagram, get_e_above_hull
using CellBase
using LaTeXStrings
using PlotlyJS


"""
Return the ternary coordinates for all stable and non stable phases 
"""
function get_ternary_hulldata(phased)
    hullbc = phased.qhull_input[1:end-1, 1:end-1]
    hulla = [(1.0 .- sum.(eachcol(hullbc)))...;;]
    hullabc = vcat(hulla, hullbc)

    # Divide into stable and unstable ones 
    stable_mask =
        map(x -> phased.min_energy_records[x] in phased.stable_records, 1:size(hulla, 2))
    unstable_mask = map(!, stable_mask)

    labels = map(x -> formula(x.composition), phased.min_energy_records)
    ehull = [phased.min_energy_e_above_hull[x] for x in phased.min_energy_records]
    (
        abc_stable=hullabc[:, stable_mask],
        labels_stable=labels[stable_mask],
        e_above_hull_stable=ehull[stable_mask],
        abc_unstable=hullabc[:, unstable_mask],
        labels_unstable=labels[unstable_mask],
        e_above_hull_unstable=ehull[unstable_mask],
        elements=phased.elements,
    )
end

"""
    make_ternary_plot(plot_data)

Compose ternary diagram from a PhaseDiagram.
"""
function make_ternary_plot(phased::PhaseDiagram)
    make_ternary_plot(get_ternary_hulldata(phased))
end

function make_ternary_plot(plot_data)

    elements = plot_data.elements
    abc_stable = plot_data.abc_stable
    labels_stable = plot_data.labels_stable
    e_above_hull_stable = plot_data.e_above_hull_stable

    abc_unstable = plot_data.abc_unstable
    labels_unstable = plot_data.labels_unstable
    e_above_hull_unstable = plot_data.e_above_hull_unstable

    function make_ax(title, tickangle)
        attr(
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

    has_unstable = length(labels_unstable) > 0
    t_stable = scatterternary(
        name="Stable",
        mode="markers",
        a=abc_stable[1, :],
        b=abc_stable[2, :],
        c=abc_stable[3, :],
        customdata=collect(zip(labels_stable)),
        e_above_hull=e_above_hull_stable,
        marker=attr(symbol="diamond", color="#60dbf1", size=14, line_width=2),
        hovertemplate="Composition: %{customdata[0]}<br><extra></extra>",
    )
    traces = [t_stable]
    if has_unstable
        t_unstable = scatterternary(
            name="Unstable",
            mode="markers",
            a=abc_unstable[1, :],
            b=abc_unstable[2, :],
            c=abc_unstable[3, :],
            customdata=collect(zip(labels_unstable, e_above_hull_unstable)),
            hovertemplate="Composition: %{customdata[0]}<br>e_above_hull: %{customdata[1]:.5f}eV/atom<br><extra></extra>",
            marker=attr(symbol="circle", color="#DB7365", size=8, line_width=0),
        )
        push!(traces, t_unstable)
    end



    layout = Layout(
        ternary=attr(
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

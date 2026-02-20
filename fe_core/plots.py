import plotly.graph_objects as go

def runwait_figure(per_driver, team_map, team_colors, title):
    order=[]
    for t,pair in team_map.items():
        order.extend(pair)
    for d in per_driver.keys():
        if d not in order:
            order.append(d)

    fig=go.Figure()
    x=list(range(len(order)))
    bar_w=0.8
    max_h=0

    for xi, drv in enumerate(order):
        if drv not in per_driver: 
            continue
        d=per_driver[drv]

        segs=[]
        for i, dur in enumerate(d['run_durs']):
            segs.append(('run', dur, d['tyre_labels'][i]))
            if i < len(d['wait_durs']):
                segs.append(('wait', d['wait_durs'][i], None))

        color='#999'
        for t,pair in team_map.items():
            if drv in pair:
                shade=0 if pair[0]==drv else 1
                color=team_colors[t][shade]
                break

        bottom=0
        for kind, dur, label in segs:
            barcolor = color if kind=='run' else '#D3D3D3'
            fig.add_trace(go.Bar(
                x=[xi], y=[dur],
                marker_color=barcolor,
                showlegend=False
            ))
            bottom+=dur
        max_h=max(max_h, bottom)

    fig.update_layout(
        title=title,
        barmode='stack',
        xaxis=dict(tickvals=x, ticktext=order, tickangle=45),
        yaxis=dict(title="Time (minutes)", range=[0,max_h*1.1]),
        height=600,
        template='plotly_white'
    )
    return fig

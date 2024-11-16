from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware 
from pydantic import BaseModel
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.interpolate import make_interp_spline
import pandas as pd
import numpy as np
from typing import List
import io

app = FastAPI()

# Enable CORS
origins = [
    "http://localhost:3000",
    "https://medinfo-backend-xie7.onrender.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # List of allowed origins
    allow_credentials=True,
    allow_methods=["POST"],  
    allow_headers=["*"],  # Allow all headers
)


class MonthlyData(BaseModel):
    month: str
    count: int
    
class PlotRequest(BaseModel):
    year: int
    data: List[MonthlyData]
    
@app.post('/plot')
async def plot_user_activities(request: PlotRequest):
    year = request.year
    input_data = request.data
        
    month_data = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sept", "Oct", "Nov", "Dec"]
    
    # Filter input data for the given year
    filtered_data = [
        {"month": data.month[:-5], "count": data.count}
        for data in input_data if data.month.endswith(str(year))
    ]
    
    # Prepare data ensuring all months are represented
    input_months = {fil_data["month"]: fil_data["count"] for fil_data in filtered_data}
    activities_count = [input_months.get(month, 0) for month in month_data]

    # Dataframe for plotting
    user_activities = pd.DataFrame({'months': month_data, 'activities_count': activities_count})

    if not user_activities["activities_count"].values.sum()<=0:
        x_smooth = np.arange(len(user_activities["activities_count"]))
        y_smooth = user_activities["activities_count"].values

        # Apply spline interpolation
        spl = make_interp_spline(x_smooth, y_smooth)  # Cubic spline
        x_smooth_new = np.linspace(x_smooth.min(), x_smooth.max(), 500)
        y_smooth_new = spl(x_smooth_new)

        # Ensure no negative values
        y_smooth_new = np.clip(y_smooth_new, 0, None)
        print(y_smooth_new.min())

        # Peak point
        peak_index = np.argmax(y_smooth_new)
        peak_x = x_smooth_new[peak_index]
        peak_y = y_smooth_new[peak_index]
        
        # Plot configuration
        plt.plot(x_smooth_new, y_smooth_new, color="#63907A", linewidth=2)
        xmin, xmax = plt.xlim()
        ymin, ymax = plt.ylim(y_smooth_new.min(), 100)

        # Create horizontal gradient
        gradient_stops = [
            (0.388, 0.565, 0.478, 1.0),
            (0.388, 0.565, 0.478, 0.0)
        ]
        cm = LinearSegmentedColormap.from_list("custom", gradient_stops)
        polygon = plt.fill_between(x_smooth_new, y_smooth_new, color="none")

        #Fill area under the curve with the custom color gradient
        gradient = plt.imshow(np.linspace(ymin, y_smooth_new.max(), 256).reshape(-1, 1), cmap=cm, interpolation="bicubic", aspect="auto", extent=[xmin, xmax, ymin, y_smooth_new.max()])
        gradient.set_clip_path(polygon.get_paths()[0], transform=plt.gca().transData)
        
        plt.scatter(peak_x, peak_y+0.5, color='#DAD7CD', s=70, zorder=7)

        # Set x-axis ticks for months
        plt.xticks(np.arange(len(month_data)), month_data)

        
    else:
        x_smooth_new = np.arange(len(month_data))
        y_smooth_new = np.zeros(len(month_data))
        
        plt.plot(x_smooth_new, y_smooth_new, linewidth=0)
        plt.ylim(y_smooth_new.min(), y_smooth_new.max()+5)

        plt.xticks(x_smooth_new, month_data)
    
    # Hide y-axis and borders
    plt.gca().axes.get_yaxis().set_visible(False)
    for spine in ['top', 'left', 'right', 'bottom']:
        plt.gca().spines[spine].set_visible(False)
    plt.tick_params(axis='x', bottom=False)
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return StreamingResponse(img, media_type='image/png')
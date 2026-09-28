import React, { useState, useEffect, useCallback } from 'react';

interface RecruiterDashboardProps {
  recruiterId: string;
}

const RecruiterDashboard: React.FC<RecruiterDashboardProps> = ({ recruiterId }) => {
  const [showSettings, setShowSettings] = useState(false);
  const [initialLoading, setInitialLoading] = useState(true);
  const [dashboardData, setDashboardData] = useState<any>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch(`/api/recruiters/${recruiterId}/dashboard`);
        const data = await response.json();
        setDashboardData(data);
        setInitialLoading(false);
      } catch (error) {
        console.error('Failed to fetch dashboard data:', error);
        setInitialLoading(false);
      }
    };
    fetchData();
  }, [recruiterId]);

  const processedData = React.useMemo(() => {
    if (!dashboardData) return null;
    return {
      ...dashboardData,
      lastUpdated: new Date().toISOString(),
    };
  }, [dashboardData]);

  if (showSettings) {
    return <div>Settings Panel</div>;
  }

  if (initialLoading) {
    return <div>Loading...</div>;
  }

  return (
    <div className="recruiter-dashboard">
      <h1>Recruiter Dashboard</h1>
      {processedData && (
        <div className="dashboard-content">
          <pre>{JSON.stringify(processedData, null, 2)}</pre>
        </div>
      )}
    </div>
  );
};

export default RecruiterDashboard;
